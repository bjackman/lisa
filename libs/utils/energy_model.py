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

from collections import namedtuple

import pandas as pd
import numpy as np

from devlib.utils.misc import memoized

ActiveState = namedtuple('ActiveState', ['capacity', 'power'])
ActiveState.__new__.__defaults__ = (None, None)

class _CpuTree(object):
    def __init__(self, cpu, children):
        if (cpu is None) == (children is None):
            raise ValueError('Provide exactly one of: cpus or children')

        if cpu is not None:
            self.cpus = [cpu]
            self.children = []
        else:
            if len(children) == 0:
                raise ValueError('children cannot be empty')
            self.cpus = sorted(set().union(*[n.cpus for n in children]))
            self.children = children
            for child in children:
                child.parent = self

        self.name = None

    def __repr__(self):
        name_bit = ''
        if self.name:
            name_bit = 'name="{}", '.format(self.name)

        if self.children:
            return '{}({}children={})'.format(
                self.__class__.__name__, name_bit, self.children)
        else:
            return '{}({}cpus={})'.format(
                self.__class__.__name__, name_bit, self.cpus)

    def _iter(self, include_non_leaves):
        for child in self.children:
            for child_i in child._iter(include_non_leaves):
                yield child_i
        if include_non_leaves or not self.children:
            yield self

    def iter_nodes(self):
        return self._iter(True)

    def iter_leaves(self):
        return self._iter(False)

class EnergyModelNode(_CpuTree):
    def __init__(self, active_states, idle_states,
                 cpu=None, children=None, name=None):
        super(EnergyModelNode, self).__init__(cpu, children)

        if cpu and not name:
            name = 'cpu' + str(cpu)

        self.name = name
        self.active_states = active_states
        self.idle_states = idle_states

    @property
    def max_capacity(self):
        return max(s.capacity for s in self.active_states.values())

    def idle_state_idx(self, state):
        return self.idle_states.keys().index(state)

class EnergyModelRoot(EnergyModelNode):
    def __init__(self, active_states=None, idle_states=None, *args, **kwargs):
        return super(EnergyModelRoot, self).__init__(
            active_states, idle_states, *args, **kwargs)

class PowerDomain(_CpuTree):
    def __init__(self, idle_states, cpu=None, children=None):
        self.idle_states = idle_states

class EnergyModel(object):
    """
    Represents hierarchical CPU topology with power and capacity data

    Describes a CPU topology similarly to trappy's Topology class, additionally
    describing relative CPU compute capacity, frequency domains and energy costs
    in various configurations.

    The topology is stored in 'levels', currently hard-coded to be 'cpu' and
    'cluster'. Each level is a list of EnergyModelNode objects. An EnergyModel
    node is a CPU or group of CPUs with associated power and (optionally)
    capacity characteristics.
    """

    # TODO check that this is the highest cap available
    capacity_scale = 1024

    def __init__(self, root_node, power_domains, freq_domains):
        self.cpus = root_node.cpus
        if self.cpus != range(len(self.cpus)):
            raise ValueError('CPUs are sparse or out of order')

        self.cpu_nodes = sorted(list(root_node.iter_leaves()),
                                key=lambda n: n.cpus[0])

    def _cpus_with_capacity(self, cap):
        return [c for c in self.cpus
                if self.cpu_nodes[c].max_capacity == cap]

    @property
    @memoized
    def biggest_cpus(self):
        max_cap = max(n.max_capacity for n in self.cpu_nodes)
        return self._cpus_with_cap(max_cap)

    @property
    @memoized
    def littlest_cpus(self):
        min_cap = min(n.max_capacity for n in self._levels[CPU_LEVEL])
        return self._cpus_with_cap(min_cap)

    @property
    @memoized
    def is_heterogeneous(self):
        """
        True iff CPUs do not all have the same efficiency and OPP range
        """
        states = self.cpu_nodes[0].active_states
        return any(c.active_states != states for c in self.cpu_nodes[1:])

    def _guess_idle_states(self, cpus_active):
        def find_deepest(pd):
            if not any(cpus_active[c] for c in pd.cpus):
                if pd.parent:
                    parent_state = find_deepest(pd.parent)
                    if parent_state:
                        return parent_state
                return pd.idle_states[-1] if len(pd.idle_states) else None
            return None

        return [find_deepest(c.power_domain) for c in self._levels["cpu"]]

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
        # TODO simplify
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
        """
        TODO DOCUMENT THIS LOL
        """
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

    # TODO this takes exponential time, we can almost certainly avoid that.
    # TODO clean up interface and document
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
