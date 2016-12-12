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
from itertools import product
import logging

import pandas as pd
import numpy as np

from devlib.utils.misc import memoized

class EnergyModelCapacityError(Exception):
    pass

ActiveState = namedtuple('ActiveState', ['capacity', 'power'])
ActiveState.__new__.__defaults__ = (None, None)

class _CpuTree(object):
    """
    Abstract representation of a CPU topology.

    Each node contains either a single CPU or a set of child nodes.
    """
    def __init__(self, cpu, children):
        if (cpu is None) == (children is None):
            raise ValueError('Provide exactly one of: cpus or children')

        self.parent = None

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
    """
    Describes topology and energy data for an EnergyModel.
    """
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
    """
    Convenience class for root of an EnergyModelNode tree.

    Just like EnergyModelNode except that active_states and idle_states aren't
    required.
    """
    def __init__(self, active_states=None, idle_states=None, *args, **kwargs):
        return super(EnergyModelRoot, self).__init__(
            active_states, idle_states, *args, **kwargs)

class PowerDomain(_CpuTree):
    """
    Describes the power domain hierarchy for an EnergyModel.

    Note that the idle_states are expected to be just a list of names - power
    data is stored elsewhere (EnergyModelNode::idle_states) using these names as
    keys. They also may be an empty list.
    """
    def __init__(self, idle_states, cpu=None, children=None):
        super(PowerDomain, self).__init__(cpu, children)
        self.idle_states = idle_states

class EnergyModel(object):
    """
    Represents hierarchical CPU topology with power and capacity data

    An energy model consists of

    - A CPU topology, representing the physical (cache/interconnect) topology of
      the CPUs.
      Each node stores the energy usage of that node's hardware when it is in
      each active or idle state. They also store a compute capacity eat each
      frequency, but this is only meaningful for leaf nodes (CPUs) and may be
      None at higher levels. These capacity values are relative; the maximum
      capacity would usually be 1024, the value of SCHED_CAPACITY_SCALE in the
      Linux kernel scheduler.
      Use EnergyModelNodes to describe this.

    - A power domain topology, representing the hierarchy of areas that can be
      powered down (idled).
      The power domains are a single tree. Leaf nodes must contain exactly one
      CPU and the root node must indirectly contain every CPU. Each power domain
      has a list (maybe empty) of names of idle states that that domain can
      enter.
      Use PowerDomains to describe this.

    - A set of frequency domains, representing groups of CPUs whose clock
      frequencies must be equal (probably because they share a clock). The
      frequency domains must be a partition of the CPUs.
    """

    capacity_scale = 1024

    def __init__(self, root_node, root_power_domain, freq_domains):
        self.cpus = root_node.cpus
        if self.cpus != range(len(self.cpus)):
            raise ValueError('CPU IDs are sparse')

        fd_intersection = set().intersection(*freq_domains)
        if fd_intersection:
            raise ValueError('CPUs {} exist in multiple freq domains'.format(
                fd_intersection))
        fd_difference = set(self.cpus) - set().union(*freq_domains)
        if fd_difference:
            raise ValueError('CPUs {} not in any frequency domain'.format(
                fd_difference))
        self.freq_domains = freq_domains

        def sorted_leaves(root):
            # Get a list of the leaf (cpu) nodes of a _CpuTree in order of the
            # CPU ID
            ret = sorted(list(root.iter_leaves()), key=lambda n: n.cpus[0])
            assert all(len(n.cpus) == 1 for n in ret)
            return ret

        self.root = root_node
        self.cpu_nodes = sorted_leaves(root_node)
        self.cpu_pds = sorted_leaves(root_power_domain)
        assert len(self.cpu_pds) == len(self.cpu_nodes)

        self._log = logging.getLogger('EnergyModel')

        max_cap = max(n.max_capacity for n in self.cpu_nodes)
        if max_cap != self.capacity_scale:
            self._log.warning(
                'Unusual max capacity (%s), overriding capacity_scale', max_cap)
            self.capacity_scale = max_cap

    def _cpus_with_capacity(self, cap):
        """
        Helper method to find the CPUs whose max capacity equals cap
        """
        return [c for c in self.cpus
                if self.cpu_nodes[c].max_capacity == cap]

    @property
    @memoized
    def biggest_cpus(self):
        """
        The CPUs with the highest compute capacity at their highest frequency
        """
        return self._cpus_with_capacity(self.capacity_scale)

    @property
    @memoized
    def littlest_cpus(self):
        """
        The CPUs with the lowest compute capacity at their highest frequency
        """
        min_cap = min(n.max_capacity for n in self.cpu_nodes)
        return self._cpus_with_capacity(min_cap)

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

        return [find_deepest(pd) for pd in self.cpu_pds]

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
        [0, 0, 0, 1] -> ["cluster-sleep-1", "cluster-sleep-1",
                         "cpu-sleep","WFI"]

        :param cpus_active: list where bool(cpus_active[N]) is False iff no
                            tasks will run on CPU N.
        """
        states = self._guess_idle_states(cpus_active)
        return [s or c.idle_states.keys()[0]
                for s, c in zip(states, self.cpu_nodes)]

    def _guess_freqs(self, util_distrib):
        overutilized = False
        # Find what frequency each CPU would need if it was alone in its
        # frequency domain
        ideal_freqs = [0 for _ in self.cpus]
        for node in self.cpu_nodes:
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

        # Rectify the frequencies among domains
        freqs = [0 for _ in ideal_freqs]
        for domain in self.freq_domains:
            domain_freq = max(ideal_freqs[c] for c in domain)
            for cpu in domain:
                freqs[cpu] = domain_freq

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

    def _estimate_from_active_time(self, cpu_active_time, freqs, idle_states,
                                   combine):
        """
        Helper for estimate_from_cpu_util

        Like estimate_from_cpu_util but uses active time i.e. proportion of time
        spent not-idle in the range 0.0 - 1.0.

        If combine=False, return idle and active power as separate components.
        """
        power = 0
        ret = {}

        assert all(0.0 <= a <= 1.0 for a in cpu_active_time)

        for node in self.root.iter_nodes():
            if not node.active_states or not node.idle_states:
                continue

            cpus = tuple(node.cpus)
            # For now we assume topology nodes with energy models do not overlap
            # with frequency domains
            freq = freqs[cpus[0]]
            assert all(freqs[c] == freq for c in cpus[1:])

            # The active time of a node is estimated as the max of the active
            # times of its children.
            # This works great for the synthetic periodic workloads we use in
            # LISA (where all threads wake up at the same time) but is probably
            # no good for real workloads.
            active_time = max(cpu_active_time[c] for c in cpus)
            active_power = node.active_states[freq].power * active_time

            _idle_power = max(node.idle_states[idle_states[c]] for c in cpus)
            idle_power = _idle_power * (1 - active_time)

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
        """
        Estimate the energy usage of the system under a utilization distribution

        Take as input a list U where U[n] is the utilization of CPU n. Assume
        that this utilization distribution is static, for example imagine each
        CPU is running a fixed periodic task that indefinitely provides the same
        amount of work.

        Optionally also take freqs; a list of frequencies at which each CPU is
        assumed to run, and idle_states, the idle states that each CPU can enter
        between activations. If not provided, they will be estimated assuming an
        ideal selection system (i.e. perfect cpufreq & cpuidle governors).

        Return a dict with power in bogo-Watts (bW), with contributions from
        each system component keyed with a tuple of the CPUs comprising that
        component, and the sum of those components keyed with 'power'. e.g:

            {
                (0,)    : 10,
                (1,)    : 10,
                (1, 2)  : 5,
                'power' : 25
            }

        This represents CPUs 0 and 1 each using 10bW and their shared
        resources using 5bW for a total of 25bW.
        """
        if freqs is None:
            freqs = self.guess_freqs(util_distrib)
        if idle_states is None:
            idle_states = self.guess_idle_states(util_distrib)

        cpu_active_time = []
        for cpu, node in enumerate(self.cpu_nodes):
            assert [cpu] == node.cpus
            cap = node.active_states[freqs[cpu]].capacity
            cpu_active_time.append(min(float(util_distrib[cpu]) / cap, 1.0))

        return self._estimate_from_active_time(cpu_active_time,
                                               freqs, idle_states, combine=True)

    def get_optimal_placements(self, capacities):
        """
        Find the optimal distribution of work for a set of tasks

        Take as input a dict mapping tasks to expected utilization values. These
        tasks are assumed not to change; they have a single static utilization
        value. A set of single-phase periodic RT-App tasks is an example of a
        suitable workload for this model.

        Returns a list of candidates which are estimated to be optimal in terms
        of power consumption, but that do not result in any CPU becoming
        over-utilized. Each candidate is a list U where U[n] is the expected
        utilization of CPU n under the task placement. Multiple task placements
        that result in the same utilization distribution are considered
        equivalent.

        If no such candidates exist, i.e. the system being modeled cannot
        satisfy the workload's throughput requirements, an
        EnergyModelCapactyError is raised. For example, if e was an EnergyModel
        modeling two CPUs with capacity 1024, this call would raise this error:

            e.get_optimal_placements({"t1": 800, "t2": 800, "t3: "800"})

        This estimation assumes an ideal system of selecting OPPs and idle
        states for CPUs.

        This is a brute force search taking time exponential wrt. the number of
        tasks.
        """
        tasks = capacities.keys()

        num_candidates = len(self.cpus) ** len(tasks)
        self._log.info(
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

        if not candidates:
            # The system can't provide full throughput to this workload.
            raise EnergyModelCapacityError(
                "Can't handle workload - total cap = {}".format(
                    sum(capacities.values())))

        # Whittle down to those that give the lowest energy estimate
        min_power = min(e["power"] for e in candidates.itervalues())
        ret = [u for u, e in candidates.iteritems() if e["power"] == min_power]

        self._log.info("%14s - Done", "EnergyModel")
        return ret
