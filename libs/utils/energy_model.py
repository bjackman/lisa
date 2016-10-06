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
