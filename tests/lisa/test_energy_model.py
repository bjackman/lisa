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

from collections import OrderedDict
import unittest
from unittest import TestCase

from energy_model import (EnergyModel, ActiveState, EnergyModelCapacityError,
                          EnergyModelNode, EnergyModelRoot, PowerDomain)

"""
A very basic test for the EnergyModel class.

Just creates a simple instance of EnergyModel describing a hypothetical
big.LITTLE system and smoke-tests the methods and properties.
"""

little_cluster_active_states = OrderedDict([
    (1000, ActiveState(power=10)),
    (1500, ActiveState(power=15)),
    (2000, ActiveState(power=20)),
])

little_cluster_idle_states = OrderedDict([
    ('WFI',              5),
    ('cpu-sleep-0',      5),
    ('cluster-sleep-0',  1),
])

little_cpu_active_states = OrderedDict([
    (1000, ActiveState(capacity=100, power=100)),
    (1500, ActiveState(capacity=150, power=150)),
    (2000, ActiveState(capacity=200, power=200)),
])

little_cpu_idle_states = OrderedDict([
    ('WFI',              5),
    ('cpu-sleep-0',      0),
    ('cluster-sleep-0',  0),
])

littles=[0, 1]
def little_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=little_cpu_active_states,
                           idle_states=little_cpu_idle_states)

big_cluster_active_states = OrderedDict([
    (3000, ActiveState(power=30)),
    (4000, ActiveState(power=40)),
])

big_cluster_idle_states = OrderedDict([
    ('WFI',              8),
    ('cpu-sleep-0',      8),
    ('cluster-sleep-0',  2),
])

big_cpu_active_states = OrderedDict([
    (3000, ActiveState(capacity=300, power=300)),
    (4000, ActiveState(capacity=400, power=400)),
])

big_cpu_idle_states = OrderedDict([
    ('WFI',              9),
    ('cpu-sleep-0',      0),
    ('cluster-sleep-0',  0),
])

bigs=[2, 3]

def big_cpu_node(cpu):
    return EnergyModelNode(cpu=cpu,
                           active_states=big_cpu_active_states,
                           idle_states=big_cpu_idle_states)

em = EnergyModel(
    root_node=EnergyModelRoot(children=[
        EnergyModelNode(name='cluster_little',
                        active_states=little_cluster_active_states,
                        idle_states=little_cluster_idle_states,
                        children=[little_cpu_node(0),
                                  little_cpu_node(1)]),
        EnergyModelNode(name='cluster_big',
                        active_states=big_cluster_active_states,
                        idle_states=big_cluster_idle_states,
                        children=[big_cpu_node(2),
                                  big_cpu_node(3)])
    ]),
    root_power_domain=PowerDomain(idle_states=[], children=[
        PowerDomain(
            idle_states=['cluster-sleep-0'],
            children=[PowerDomain(idle_states=['WFI', 'cpu-sleep-0'], cpu=c)
                      for c in littles]),
        PowerDomain(
            idle_states=['cluster-sleep-0'],
            children=[PowerDomain(idle_states=['WFI', 'cpu-sleep-0'], cpu=c)
                      for c in bigs]),
        ]),
    freq_domains=[littles, bigs]
)

class TestFlatten(TestCase):
    def test_flatten_energy(self):
        flattened = em.flatten_energy()
        for node in flattened.root.iter_nodes():
            if node.children:
                # Non-leaf node. Any energy data should be zero
                active_states = node.active_states or {}
                self.assertTrue(
                    all(s.power == 0 for s in active_states.values()))
                idle_states = node.idle_states or {}
                self.assertTrue(
                    all(p == 0 for p in idle_states.values()))

        leaf_idle_power = {}
        for state in flattened.cpu_nodes[0].idle_states:
            leaf_idle_power[state] = [n.idle_states[state]
                                      for n in flattened.cpu_nodes]

        self.assertDictEqual(leaf_idle_power, {
            'WFI' :            [7.5, 7.5, 13.0, 13.0],
            'cpu-sleep-0':     [2.5, 2.5, 4.0, 4.0],
            'cluster-sleep-0': [0.5, 0.5, 1.0, 1.0],
        })

        self.assertDictEqual(flattened.cpu_nodes[0].active_states, OrderedDict([
            (1000, ActiveState(capacity=100, power=105.0)),
            (1500, ActiveState(capacity=150, power=157.5)),
            (2000, ActiveState(capacity=200, power=210.0)),
        ]))
        self.assertDictEqual(flattened.cpu_nodes[0].active_states,
                             flattened.cpu_nodes[1].active_states)

        self.assertDictEqual(flattened.cpu_nodes[2].active_states, OrderedDict([
            (3000, ActiveState(capacity=300, power=315)),
            (4000, ActiveState(capacity=400, power=420)),
        ]))
        self.assertDictEqual(flattened.cpu_nodes[2].active_states,
                             flattened.cpu_nodes[3].active_states)

    def test_flatten_power_domains(self):
        flattened = em.flatten_power_domains()
        self.assertListEqual(flattened.root_pd.idle_states, [])
        self.assertEqual(len(flattened.root_pd.children), 4)
        for pd in flattened.root_pd.children:
            self.assertListEqual(pd.children, [])
            self.assertListEqual(pd.idle_states,
                                 ['WFI', 'cpu-sleep-0', 'cluster-sleep-0'])


class TestOptimalPlacement(TestCase):
    def assertPlacementListEqual(self, l1, l2):
        """
        Assert that a pair of lists of lists contain the same lists in any order
        """
        s1 = set([tuple(l) for l in l1])
        s2 = set([tuple(l) for l in l2])
        self.assertSetEqual(s1, s2)

    def test_single_small(self):
        placements = em.get_optimal_placements({'task0': 1})
        self.assertPlacementListEqual(placements, [[1, 0, 0, 0],
                                                   [0, 1, 0, 0]])

    def test_single_big(self):
        placements = em.get_optimal_placements({'task0': 350})
        self.assertPlacementListEqual(placements, [[0, 0, 350, 0],
                                                   [0, 0, 0, 350]])

    def test_packing(self):
        tasks = {'task' + str(i) : 10 for i in range(5)}
        placements = em.get_optimal_placements(tasks)
        total_util = sum(tasks.values())
        self.assertPlacementListEqual(placements, [[total_util, 0, 0, 0],
                                                   [0, total_util, 0, 0]])

    def test_overutilized_single(self):
        self.assertRaises(EnergyModelCapacityError,
                          em.get_optimal_placements, {'task0' : 401})

    def test_overutilized_many(self):
        total_cap = 400 * 2 + 200 * 2
        task_size = 200
        tasks = {'task' + str(i): task_size
                 for i in range((total_cap / task_size) + 1)}
        self.assertRaises(EnergyModelCapacityError,
                          em.get_optimal_placements, tasks)

class TestBiggestCpus(TestCase):
    def test_biggest_cpus(self):
        self.assertEqual(em.biggest_cpus, [2, 3])

class TestLittlestCpus(TestCase):
    def test_littlest_cpus(self):
        self.assertEqual(em.littlest_cpus, [0, 1])

class TestMaxCap(TestCase):
    def test_max_cap(self):
        max_caps = [n.max_capacity for n in em.cpu_nodes]
        self.assertEqual(max_caps, [200, 200, 400, 400])

class TestEnergyEst(TestCase):
    def test_all_overutilized(self):
        big_cpu = 400 * 2
        little_cpu =  200 * 2
        big_cluster = 40
        little_cluster = 20

        total = big_cpu + little_cpu + big_cluster + little_cluster

        power = em.estimate_from_cpu_util([10000] * 4)
        exp = {
            'power' : total,
            (0): { 'active': little_cpu, 'idle': 0},
            (1): { 'active': little_cpu, 'idle': 0},
            (2): { 'active': big_cpu, 'idle': 0},
            (3): { 'active': big_cpu, 'idle': 0},
            (0, 1): { 'active': little_cluster, 'idle': 0},
            (2, 3): { 'active': big_cluster, 'idle': 0}
        }
        for k, v in power.iteritems():
            self.assertAlmostEqual(v, power[k])

    def test_all_idle(self):
        self.assertEqual(em.estimate_from_cpu_util([0, 0, 0, 0])['power'],
                         0 * 4 # CPU power = 0
                         + 2   # big cluster power
                         + 1)  # LITTLE cluster power

    def test_one_little_half_lowest(self):
        cpu0_util = 100 * 0.5
        self.assertEqual(em.estimate_from_cpu_util([cpu0_util, 0, 0, 0])['power'],
                         (0.5 * 100)  # CPU0 active power
                         + (0.5 * 5)  # CPU0 idle power
                         + (0.5 * 5)  # LITTLE cluster idle power
                         + (0.5 * 10) # LITTLE cluster active power
                         + 2)         # big cluster power

class TestIdleStates(TestCase):
    def test_zero_util_deepest(self):
        self.assertEqual(em.guess_idle_states([0] * 4), ['cluster-sleep-0'] * 4)

    def test_single_cpu_used(self):
        states = em.guess_idle_states([0, 0, 0, 1])
        self.assertEqual(states, ['cluster-sleep-0', 'cluster-sleep-0',
                                  'cpu-sleep-0', 'WFI'])

        states = em.guess_idle_states([0, 1, 0, 0])
        self.assertEqual(states, ['cpu-sleep-0', 'WFI',
                                  'cluster-sleep-0', 'cluster-sleep-0',])

    def test_all_cpus_used(self):
        states = em.guess_idle_states([1, 1, 1, 1])
        self.assertEqual(states, ['WFI'] * 4)

    def test_one_cpu_per_cluster(self):
        states = em.guess_idle_states([0, 1, 0, 1])
        self.assertEqual(states, ['cpu-sleep-0', 'WFI'] * 2)

class TestFreqs(TestCase):

    def test_zero_util_slowest(self):
        self.assertEqual(em.guess_freqs([0] * 4),
                         [1000, 1000, 3000, 3000])

    def test_high_util_fastest(self):
        self.assertEqual(em.guess_freqs([100000] * 4),
                         [2000, 2000, 4000, 4000])

    def test_freq_domains(self):
        self.assertEqual(em.guess_freqs([0, 0, 0, 10000]),
                         [1000, 1000, 4000, 4000])

        self.assertEqual(em.guess_freqs([0, 10000, 0, 10000]),
                         [2000, 2000, 4000, 4000])

        self.assertEqual(em.guess_freqs([0, 10000, 0, 0]),
                         [2000, 2000, 3000, 3000])

    def test_middle_freq(self):
        self.assertEqual(em.guess_freqs([0, 110, 0, 0]),
                         [1500, 1500, 3000, 3000])
