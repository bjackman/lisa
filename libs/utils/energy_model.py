# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2016, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from collections import namedtuple, OrderedDict
from itertools import product

# TODO get Topology out of here and share it across Lisa
#

import pandas as pd
import numpy as np

from bart.common.Utils import interval_sum
from devlib import TargetError

ActiveState = namedtuple("ActiveState", ["capacity", "power"])
ActiveState.__new__.__defaults__ = (None, None)

class EnergyModelNode(namedtuple("EnergyModelNode",
                                 ["cpus", "active_states", "idle_states",
                                  "power_domain", "freq_domain"])):
    @property
    def max_capacity(self):
        return max(s.capacity for s in self.active_states.values())

    def idle_state_idx(self, state):
        return self.idle_states.keys().index(state)

EnergyModelNode.__new__.__defaults__ = (None, None, None, None, None)

class PowerDomain(object):
    def __init__(self, idle_states, parent, cpus):
        self.cpus = set(cpus)
        self.idle_states = idle_states
        self.children = []

        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
            self.parent.cpus = self.parent.cpus.union(self.cpus)

    def __repr__(self):
        return "PowerDomain(cpus={})".format(list(self.cpus))

class EnergyModel(object):
    capacity_scale = 1024

    def __init__(self, target=None, levels=None):

        if target:
            raise NotImplementedError()

        self._levels = levels

        self.num_cpus = len(self._levels["cpu"])
        self.cpus = [n.cpus[0] for n in levels["cpu"]]
        if self.cpus != range(self.num_cpus):
            raise ValueError("CPUs are sparse or out of order")
        if any(len(n.cpus) != 1 for n in levels["cpu"]):
            raise ValueError("'cpu' level nodes must all have exactly 1 CPU")

    def get_level(self, level_name):
        return self._levels[level_name]

    def _get_from_target(self, target):
        topology = self.topology

        sched_domain_path = "/proc/sys/kernel/sched_domain/"

        # Check there's an "MC" sched_domain_topology_level and it's the bottom.
        domain0_name_path = sched_domain_path + "/cpu0/domain0/name"
        domain0_name = target.read_value(domain0_name_path)
        if domain0_name != "MC":
            # Probably CONFIG_SCHED_MC is disabled on the target or there's an
            # SMT sched_domain. This won't work in either of those situations.
            raise TargetError(
                "The lowest sched_domain is {}, not 'MC'".format(domain0_name))

        # Currently hard-coded for two levels, "cpu" and "cluster", but should
        # in theory be extensible to arbitrary depth.
        levels = {"cpu": [], "cluster": []}
        power_domains = []
        freq_domains = []

        # Assume a CPU always occurs in its own group 0
        nrg_dir_fmt = "{}/cpu{{}}/domain{{}}/group0/energy/".format(
            sched_domain_path)

        def parse_active_states(cpu, sched_domain):
            path = nrg_dir_fmt.format(cpu, sched_domain) + "cap_states"
            vals = [int(v) for v in target.read_value(path).split()]
            # cap_states file is a list of (capacity, power) pairs
            cap_states = [ActiveState(c, p)
                          for c, p in zip(vals[::2], vals[1::2])]

            freqs = target.cpufreq.list_frequencies(cpu)

            assert sorted(cap_states, key=lambda s: s.capacity) == cap_states
            assert sorted(freqs) == freqs
            assert len(freqs) == len(cap_states)

            return OrderedDict([(freq, state) for freq, state
                                in zip(freqs, cap_states)])

        def parse_idle_states(cpu, domain):
            state_names = [s.name for s in target.cpuidle.get_states(cpu)]

            # cpuidle sysfs has a "power" member but it isn't initialized, use
            # the sched_group_energy data to get idle state power usage.
            path = nrg_dir_fmt.format(cpu, domain) + "idle_states"
            _power_nums = [int(v) for v in target.read_value(path).split()]
            # The first entry in the idle_states array is an implementation
            # detail
            power_nums = _power_nums[1:]

            assert len(power_nums) == len(state_names)

            return OrderedDict([(n, p) for n, p in zip(state_names, power_nums)])

        for cpus in topology.get_level("cluster"):
            cpu = cpus[0]

            #
            # Read CPU level data for this cluster
            #

            # To save time, for now we'll assume that all CPUs in the cluster
            # have the same idle and active states
            active_states = parse_active_states(cpu, 0)
            idle_states = parse_idle_states(cpu, 0)

            for cpu in cpus:
                # Linux doesn't have topological idle state information so we'll
                # assume that every idle state can be entered by every CPU
                # independently. This is _not true in real life_ but it is the
                # assumption made by the kernel.
                # (i.e. the below is commented out because we don't assume any
                #  power domains)
                # pd = PowerDomain([cpu], idle_states.values())
                # power_domains.append(pd)

                node = EnergyModelNode([cpu], active_states, idle_states)
                levels["cpu"].append(node)

                if not any(cpu in d for d in freq_domains):
                    freq_domains.append(target.cpufreq.get_domain_cpus(cpu))

            #
            # Read cluster-level data
            #

            active_states = parse_active_states(cpu, 1)
            idle_states = parse_idle_states(cpu, 1)

            node = EnergyModelNode(cpus, active_states, idle_states)
            levels["cluster"].append(node)

        return levels, power_domains, freq_domains

    def _guess_idle_states(self, util_distrib):
        def find_deepest(pd):
            if not any(util_distrib[c] != 0 for c in pd.cpus):
                if pd.parent:
                    parent_state = find_deepest(pd.parent)
                    if parent_state:
                        return parent_state
                return pd.idle_states[-1] if len(pd.idle_states) else None
            return None

        return [find_deepest(c.power_domain) for c in self._levels["cpu"]]

    # TODO clean up this dict/list state addressing mess...
    # I think we can probably go back to just dicts, the only reason we need the
    # idxs is for "min" right? Maybe we can just get "shallowest", or even give
    # idle states a compare biz

    def _guess_idle_idxs(self, util_distrib):
        states = self._guess_idle_states(util_distrib)
        return [c.idle_state_idx(s) if s else -1
                for s, c in zip(states, self._levels["cpu"])]

    def guess_idle_states(self, util_distrib):
        states = self._guess_idle_states(util_distrib)
        return [s or c.idle_states.keys()[0]
                for s, c in zip(states, self._levels["cpu"])]

    def _guess_freqs(self, util_distrib):
        overutilized = False

        # Find what frequency each CPU would need if it was alone in its
        # frequency domain
        ideal_freqs = [0 for _ in range(self.num_cpus)]
        for node in self._levels["cpu"]:
            [cpu] = node.cpus
            required_cap = util_distrib[cpu]

            possible_freqs = [f for f, s in node.active_states.iteritems()
                              if s.capacity >= required_cap]

            if possible_freqs:
                ideal_freqs[cpu] = min(possible_freqs)
            else:
                # CPU cannot provide required capacity, use max freq
                ideal_freqs[cpu] = max(node.active_states.keys())
                overutilized = True

        freqs = [0 for _ in ideal_freqs]
        for node in self._levels["cpu"]:
            [cpu] = node.cpus

            # Each CPU has to run at the max frequency required by any in its
            # frequency domain
            freq_domain = node.freq_domain
            freqs[cpu] = max(ideal_freqs[c] for c in freq_domain)

        return freqs, overutilized

    def guess_freqs(self, util_distrib):
        freqs, _ = self._guess_freqs(util_distrib)
        return freqs

    def would_overutilize(self, util_distrib):
        _, overutilized = self._guess_freqs(util_distrib)
        return overutilized

    def _estimate_from_active_time(self, cpu_active_time, freqs, idle_states,
                                   combine=False):
        power = 0

        ret = {}

        for cpu, node in enumerate(self._levels["cpu"]):
            assert [cpu] == node.cpus

            active_power = (node.active_states[freqs[cpu]].power
                            * cpu_active_time[cpu])
            idle_power = (node.idle_states[idle_states[cpu]]
                          * (1 - cpu_active_time[cpu]))

            if combine:
                ret[(cpu,)] = active_power + idle_power
            else:
                ret[(cpu,)] = {}
                ret[(cpu,)]["active"] = active_power
                ret[(cpu,)]["idle"] = idle_power

            power += active_power + idle_power

        for node in self._levels["cluster"]:
            cpus = tuple(node.cpus)

            # For now we assume clusters map to frequency domains 1:1
            freq = freqs[cpus[0]]
            assert all(freqs[c] == freq for c in cpus[1:])

            # This works great for the synthetic periodic workloads we use in
            # Lisa (where all threads wake up at the same time) but is no good
            # for real workloads.
            active_time = max(cpu_active_time[c] for c in cpus)

            active_power = node.active_states[freq].power * active_time
            idle_power = (max([node.idle_states[idle_states[c]] for c in cpus])
                          * (1 - active_time))

            if combine:
                ret[cpus] = active_power + idle_power
            else:
                ret[cpus] = {}
                ret[cpus]["active"] = active_power
                ret[cpus]["idle"] = idle_power

            power += active_power + idle_power

        ret["power"] = power
        return ret

    def estimate_from_cpu_util(self, util_distrib, freqs=None, idle_states=None):
        if freqs is None:
            freqs = self.guess_freqs(util_distrib)
        if idle_states is None:
            idle_states = self.guess_idle_states(util_distrib)

        power = 0

        cpu_active_time = []
        for cpu, node in enumerate(self._levels["cpu"]):
            assert [cpu] == node.cpus
            cap = node.active_states[freqs[cpu]].capacity
            cpu_active_time.append(min(float(util_distrib[cpu]) / cap, 1.0))

        return self._estimate_from_active_time(cpu_active_time,
                                               freqs, idle_states)

    @property
    def biggest_cpus(self):
        max_cap = max(n.max_capacity for n in self.get_level("cpu"))
        return [n.cpus[0] for n in self.get_level("cpu")
                if n.max_capacity == max_cap]

    # TODO: Clean up energy vs power in here

    # TODO: DOCUMENT AND COMMENT THIS MESS

    # TODO this takes exponential time, we can almost certainly avoid that.
    def _find_optimal_placements(self, capacities):
        tasks = capacities.keys()

        num_candidates = len(self.cpus) ** len(tasks)
        logging.info(
            "%14s - Searching %d configurations for optimal task placement...",
            "EnergyModel", num_candidates)

        candidates = {}
        for cpus in product(self.cpus, repeat=len(tasks)):
            placement = {task: cpu for task, cpu in zip(tasks, cpus)}

            util = [0 for _ in self.cpus]
            for task, cpu in placement.items():
                util[cpu] += capacities[task]
            util = tuple(util)

            if util not in candidates:
                freqs, overutilized = self._guess_freqs(util)
                if not overutilized:
                    power = self.estimate_from_cpu_util(util, freqs=freqs)
                    candidates[util] = power

        # Whittle down to those that give the lowest energy estimate
        min_power = min(e["power"] for e in candidates.itervalues())

        logging.info("%14s - Done", "EnergyModel")
        return min_power, [u for u, e in candidates.iteritems() if e["power"] == min_power]

    def estimate_workload_power(self, capacities):
        power, utils = self._find_optimal_placements(capacities)
        return power

    def reconcile_freqs(self, freqs):
        """Take a list of frequencies and make frequencies consistent in domains

        Take a list of N frequencies where freqs[N] provides the frequency of
        CPU N and return a version where whenever CPUs N and M share a frequency
        domain, freqs[N] == freqs[M]. The order of preference for which
        frequencies are changed to achieve this is unspecified.
        """

        if len(freqs) != len(self.cpus):
            raise ValueError("Bad length for frequency list")
        # We're going to mutate these lists so make copies (hence `list()`)
        remaining_cpus = list(self.cpus)
        freqs = list(freqs)
        while remaining_cpus:
            cpu = remaining_cpus[0]
            freq_domain = self._levels["cpu"][cpu].freq_domain
            for domain_cpu in freq_domain:
                freqs[domain_cpu] = freqs[cpu]
                remaining_cpus.remove(domain_cpu)

        return freqs

    # TODO: We need a "power in state" function that takes a list of frequencies
    # and idle states and returns power, then use that function throughout.

    # TODO: The function below takes several seconds, I bet it can be made
    # faster with Ninjutsu

    def estimate_from_trace(self, trace):
        # TODO all this DataFrame mangling makes Pandas complain; we might be
        # aliasing things we shouldn't be.
        idle_df = trace.ftrace.cpu_idle.data_frame
        if idle_df.empty:
            raise ValueError("No cpu_idle events found")
        idle_df = idle_df.pivot(columns="cpu_id")["state"]
        idle_df.fillna(method="ffill", inplace=True)

        freq_df = trace.ftrace.cpu_frequency.data_frame
        if freq_df.empty:
            raise ValueError("No cpu_frequency events found")
        freq_df = freq_df.pivot(columns="cpu")["frequency"]
        freq_df.fillna(method="ffill", inplace=True)

        df = pd.concat([idle_df, freq_df], axis=1, keys=["idle", "freq"])
        df.fillna(method="ffill", inplace=True)

        # Where we don't have the data (because no events arrived yet), fill it
        # in with worst-case.
        max_freqs = {c: max(n.active_states.keys())
                     for c, n in enumerate(self._levels["cpu"])}
        df.fillna({"idle": -1, "freq": max_freqs}, inplace=True)

        cpu_columns = [(c,) for c in self.cpus]
        cluster_columns = [tuple(n.cpus) for n in self._levels["cluster"]]
        columns = ["power"] + cpu_columns + cluster_columns

        def row_power(row):
            # Pandas converts our numbers to floats so it can use NaN, convert
            # them back to ints.
            def to_int(i):
                if pd.isnull(i):
                    return -1
                else:
                    return int(i)

            # These are the states cpuidle thinks the CPUs are in
            cpuidle_idxs = [to_int(i) for i in row["idle"].values]

            # These are the deepest states the hardware could actually enter
            util_distrib = [int(s < 0) for s in cpuidle_idxs]
            ideal_idxs = self._guess_idle_idxs(util_distrib)

            # The states the HW actually enters is generally the shallowest of
            # the two possibilities above.
            idle_idxs = [min(c, i) for c, i in zip(cpuidle_idxs, ideal_idxs)]
            # Note that where the idle state index is -1, this will give an
            # invalid state name. However the idle power will be 0 anyway since
            # the CPU is active.
            idle_states = [n.idle_states.keys()[i] for n, i
                           in zip(self._levels["cpu"], idle_idxs)]

            # Make frequency events consistent with HW frequency domains.
            # Linux knows about frequency domains and will trace all the
            # frequencies correctly, but the different CPU freqs are traced in
            # different ftrace events, so there is a short period where the
            # traced frequencies are inconsistent after a frequency change.
            freqs = self.reconcile_freqs([row["freq"][cpu] for cpu in self.cpus])

            power = self._estimate_from_active_time(util_distrib,
                                                    idle_states=idle_states,
                                                    freqs=freqs,
                                                    combine=True)

            return pd.Series([power[k] for k in columns], index=columns)

        logging.info("%14s - Estimating energy from trace - %d events...",
                     "EnergyModel", len(df))

        power_df = df.apply(row_power, axis=1)

        logging.info("%14s - Done.", "EnergyModel")
        return power_df

    @classmethod
    def get_power_histogram(cls, power_df):
        power_vals = power_df["power"].unique()
        time_at_power = [interval_sum(power_df["power"], value=p)
                         for p in power_vals]
        histogram = pd.DataFrame(
            time_at_power, index=power_vals, columns=["power"])
        histogram.sort_index(inplace=True)

        return histogram
