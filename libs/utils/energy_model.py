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
# from trappy.stats.Topology import Topology

from devlib import TargetError

ActiveState = namedtuple("ActiveState", ["capacity", "power"])
ActiveState.__new__.__defaults__ = (None, None)

EnergyModelNode = namedtuple(
    "EnergyModelNode", ["cpus", "active_states", "idle_states",
                        "power_domain", "freq_domain"])
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

class EnergyModel(object):
    capacity_scale = 1024

    def __init__(self, target=None, levels=None):

        if target:
            raise NotImplementedError()

        self._levels = levels

        self.num_cpus = len(self._levels["cpu"])

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

        # for level_idx, level_name in enumerate(topology):
        #     topo_level = topology.get_level(level_name)

    def guess_idle_states(self, util_distrib):
        def find_deepest(pd):
            if not any(util_distrib[c] != 0 for c in pd.cpus):
                if pd.parent:
                    parent_state = find_deepest(pd.parent)
                    if parent_state:
                        return parent_state
                return pd.idle_states[-1]
            return None

        return [find_deepest(c.power_domain) or c.power_domain.idle_states[0]
                for c in self._levels["cpu"]]

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

    def estimate_from_cpu_util(self, util_distrib):
        freqs = self.guess_freqs(util_distrib)
        idle_states = self.guess_idle_states(util_distrib)

        power = 0

        cpu_active_time = []
        for cpu, node in enumerate(self._levels["cpu"]):
            assert [cpu] == node.cpus

            cap = node.active_states[freqs[cpu]].power
            cpu_active_time.append(min(util_distrib[cpu] / cap, 1.0))

            active_power = node.active_states[freqs[cpu]].power
            idle_power = node.idle_states[idle_states[cpu]]

            power += (cpu_active_time[cpu] * active_power
                      + ((1 - cpu_active_time[cpu]) * idle_power))

        for node in self._levels["cluster"]:
            cpus = node.cpus

            freq = freqs[cpus[0]]
            idle_state = idle_states[cpus[0]]

            # For now we assume clusters map to frequency domains 1:1
            assert all(freqs[c] == freq for c in cpus[1:])

            def mean(xs):
                return sum(xs) / len(xs)

            active_power = node.active_states[freq].power
            idle_power = mean([node.idle_states[idle_states[c]] for c in cpus])

            # This works great for the synthetic periodic workloads we use in
            # Lisa (where all threads wake up at the same time) but is no good
            # for real workloads.
            active_time = max(cpu_active_time[c] for c in cpus)

            power += (active_time * active_power
                       + ((1 - active_time) * idle_power))

        return power
