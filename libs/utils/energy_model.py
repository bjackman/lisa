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
    def __init__(self, cpus, children):
        if (cpus is None) == (children is None):
            raise ValueError('Provide exactly one of: cpus or children')

        if cpus is not None:
            if len(cpus) == 0:
                raise ValueError('cpus cannot be empty')
            self.cpus = cpus
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
                 cpus=None, children=None, name=None):
        super(EnergyModelNode, self).__init__(cpus, children)

        if not self.children:
            if len(cpus) != 1:
                raise ValueError('Leaf EnergyModelNodes must have a single CPU')
            if not name:
                name = 'cpu' + str(cpus[0])

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
    def __init__(self, idle_states, cpus=None, children=None):
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
