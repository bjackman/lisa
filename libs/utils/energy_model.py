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
        self.cpus = set()
        self.idle_states = idle_states

        self.parent = parent
        self.add_cpus(set(cpus))

    def add_cpus(self, cpus):
        self.cpus = self.cpus.union(cpus)
        if self.parent:
            self.parent.add_cpus(self.cpus)

    def __repr__(self):
        return "PowerDomain(cpus={})".format(list(self.cpus))

class EnergyModel(object):
    """
    Represents hierarchical CPU topology with power and capacity data

    Describes a CPU topology similarly to trappy's Topology class, additionally
    describing relative CPU compute capacity, frequency domains and energy costs
    in various configurations.

    The topology is stored in "levels", currently hard-coded to be "cpu" and
    "cluster". Each level is a list of EnergyModelNode objects. An EnergyModel
    node is a CPU or group of CPUs with associated power and (optionally)
    capacity characteristics.
    """

    # TODO check that inputs match this
    capacity_scale = 1024

    def __init__(self, levels=None):
        self._levels = levels

        self.num_cpus = len(self._levels["cpu"])
        self.cpus = [n.cpus[0] for n in levels["cpu"]]
        if self.cpus != range(self.num_cpus):
            raise ValueError("CPUs are sparse or out of order")
        if any(len(n.cpus) != 1 for n in levels["cpu"]):
            raise ValueError("'cpu' level nodes must all have exactly 1 CPU")

    def get_level(self, level_name):
        return self._levels[level_name]

    def _guess_idle_states(self, cpus_active):
        def find_deepest(pd):
            if not any(cpus_active[c] != 0 for c in pd.cpus):
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

    def _guess_idle_idxs(self, cpus_active):
        states = self._guess_idle_states(cpus_active)
        return [c.idle_state_idx(s) if s else -1
                for s, c in zip(states, self._levels["cpu"])]

    def guess_idle_states(self, cpus_active):
        """Pessimistically guess the idle states that each CPU may enter

        If a CPU has any tasks it is estimated that it may only enter its
        shallowest idle state in between task activations. If all the CPUs
        within a power domain have no tasks, they will all be judged able to
        enter that domain's deepest idle state. If any CPU in a domain has work,
        no CPUs in that domain are assumed to enter any domain shared state.

        e.g. Consider a system with
        - two power domains PD0 and PD1
        - 4 CPUs, with CPUs [0, 1] in PD0 and CPUs [2, 3] in PD1
        - 4 idle states: "WFI", "cpu-sleep", "cluster-sleep-0" and
          "cluster-sleep-1"

        Then here are some example inputs and outputs:

        # All CPUs idle:
        [0, 0, 0, 0] -> ["cluster-sleep-0", "cluster-sleep-0",
                         "cluster-sleep-0", "cluster-sleep-0"]

        # All CPUs have work
        [1, 1, 1, 1] -> ["WFI","WFI","WFI", "WFI"]

        # One power domain active, the other idle
        [0, 0, 1, 1] -> ["cluster-sleep-1", "cluster-sleep-1", "WFI","WFI"]

        # One CPU active.
        # Note that CPU 2 has no work but is assumed to never be able to enter
        # any "cluster" state.
        [0, 0, 0, 1] -> ["cluster-sleep-1", "cpu-sleep", "WFI","WFI"]

        :param cpus_active: list where bool(cpus_active[N]) is False iff no
        tasks will run on CPU N.
        """
        states = self._guess_idle_states(cpus_active)
        return [s or c.idle_states.keys()[0]
                for s, c in zip(states, self._levels["cpu"])]

    def _guess_freqs(self, util_distrib):
        overutilized = False

        # TODO would be simpler to iter over domains and set all CPUs. Need to
        # store the set of domains somewhere.

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
        """
        Work out CPU frequencies required to execute a workload

        The input is a list where util_distrib[N] is the sum of the
        frequency-invariant, capacity-invariant utilization of tasks placed CPU
        N. That is, the quantity represented by util_avg in the Linux kernel's
        per-entity load-tracking (PELT) system.

        The range of utilization values is 0 - EnergyModel.capacity_scale, where
        EnergyModel.capacity_scale represents the computational capacity of the
        most powerful CPU at its highest available frequency.

        This function returns the lowest possible frequency for each CPU that
        provides enough capacity to satisfy the utilization, taking into account
        frequency domains.
        """
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

    def estimate_from_cpu_util(self, util_distrib, freqs=None, idle_states=None,
                               combine=False):
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
                                               freqs, idle_states, combine)

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
        def get_df(event, cpu_field, data_field):
            df = getattr(trace.ftrace, event).data_frame
            df = df.pivot(columns=cpu_field).fillna(method="ffill")
            return df[data_field]

        idle_df = get_df("cpu_idle", "cpu_id", "state")
        # If there were no idle events at all on a CPU, assume it was deeply
        # idle the whole time. This won't generally be true from a hardware
        # perspective (due to shared idle states) but should be true from
        # cpuidle's perspective.
        # TODO we check what happens if there were no events for any CPU
        #      (hopefully get_df fails)
        for cpu, node in enumerate(self._levels["cpu"]):
            if cpu not in idle_df.columns:
                deepest_idx = len(node.idle_states) - 1
                idle_df.loc[:, cpu] = pd.Series(
                    (deepest_idx for _ in idle_df.index), index=idle_df.index)

        freq_df = get_df("cpu_frequency", "cpu", "frequency")

        df = pd.concat([idle_df, freq_df], axis=1, keys=["idle", "freq"])
        df = df.fillna(method="ffill")

        # The fillna call below causes a SettingWithCopyWarning which is
        # spurious. Disable it, re-enable afterwards.
        chained_assignment = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None

        # Where we don't have the data (because no events arrived yet), fill it
        # in with worst-case.
        max_freqs = {c: max(n.active_states.keys())
                     for c, n in enumerate(self._levels["cpu"])}
        df = df.fillna({"idle": -1, "freq": max_freqs})

        pd.options.mode.chained_assignment = chained_assignment

        cpu_columns = [(c,) for c in self.cpus]
        cluster_columns = [tuple(n.cpus) for n in self._levels["cluster"]]
        out_columns = ["power"] + cpu_columns + cluster_columns

        power_memo = {}

        def row_power(row):
            # The energy estimation code eats and drinks CPU time with great
            # merriness and mirth, so don't call it more than necessary.
            memo_key = tuple(row)
            try:
                return power_memo[memo_key]
            except KeyError:
                pass

            # These are the states cpuidle thinks the CPUs are in
            cpuidle_idxs = [int(i) for i in row["idle"].values]

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

            ret = pd.Series([power[k] for k in out_columns], index=out_columns)
            power_memo[memo_key] = ret
            return ret

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
