from energy_model import (EnergyModel,
                          ActiveState, EnergyModelNode, PowerDomain)
from unittest import TestCase

little_cluster_active_states = {
    1000: ActiveState(power=10),
    2000: ActiveState(power=20),
}

little_cluster_idle_states = {
    "WFI":              5,
    "cpu-sleep-0":      5,
    "cluster-sleep-0":  1
}

little_cpu_active_states = {
    1000: ActiveState(capacity=100, power=100),
    1500: ActiveState(capacity=150, power=150),
    2000: ActiveState(capacity=200, power=200),
}

little_cpu_idle_states = {
    "WFI":              5,
    "cpu-sleep-0":      0,
    "cluster-sleep-0":  0
}

littles=[0, 1]
little_pd = PowerDomain(cpus=littles,
                        idle_states=["cluster-sleep-0"],
                        parent=None)

def little_cpu_node(cpu):
    cpu_pd = PowerDomain(cpus=[cpu],
                         idle_states=["WFI", "cpu-sleep-0"],
                         parent=little_pd)

    return EnergyModelNode([cpu],
                           active_states=little_cpu_active_states,
                           idle_states=little_cpu_idle_states,
                           power_domain=cpu_pd,
                           freq_domain=littles)

big_cluster_active_states = {
    3000: ActiveState(power=30),
    4000: ActiveState(power=40),
}

big_cluster_idle_states = {
    "WFI":              8,
    "cpu-sleep-0":      8,
    "cluster-sleep-0":  2
}

big_cpu_active_states = {
    3000: ActiveState(capacity=300, power=300),
    4000: ActiveState(capacity=400, power=400),
}

big_cpu_idle_states = {
    "WFI":              9,
    "cpu-sleep-0":      0,
    "cluster-sleep-0":  0
}

bigs=[2, 3]
big_pd = PowerDomain(cpus=bigs,
                     idle_states=["cluster-sleep-0"],
                     parent=None)

def big_cpu_node(cpu):
    cpu_pd = PowerDomain(cpus=[cpu],
                         idle_states=["WFI", "cpu-sleep-0"],
                         parent=big_pd)

    return EnergyModelNode([cpu],
                           active_states=big_cpu_active_states,
                           idle_states=big_cpu_idle_states,
                           power_domain=cpu_pd,
                           freq_domain=bigs)

levels = {
    "cluster": [
        EnergyModelNode(cpus=bigs,
                        active_states=big_cluster_active_states,
                        idle_states=big_cluster_idle_states),
        EnergyModelNode(cpus=littles,
                        active_states=little_cluster_active_states,
                        idle_states=little_cluster_idle_states)
    ],
    "cpu": [little_cpu_node(0), little_cpu_node(1),
            big_cpu_node(2), big_cpu_node(3)]
}

em = EnergyModel(levels=levels)

class TestEnergyEst(TestCase):
    def test_all_overutilized(self):
        self.assertEqual(em.estimate_from_cpu_util([10000] * 4),
                         400 * 2    # big CPU power
                         + 200 * 2  # LITTLE CPU power
                         + 40       # big cluster power
                         + 20)      # LITTLE cluster power

    def test_all_idle(self):
        self.assertEqual(em.estimate_from_cpu_util([0, 0, 0, 0]),
                         0 * 4 # CPU power = 0
                         + 2   # big cluster power
                         + 1)  # LITTLE cluster power

class TestIdleStates(TestCase):
    def test_zero_util_deepest(self):
        self.assertEqual(em.guess_idle_states([0] * 4), ["cluster-sleep-0"] * 4)

    def test_single_cpu_used(self):
        states = em.guess_idle_states([0, 0, 0, 1])
        self.assertEqual(states, ["cluster-sleep-0", "cluster-sleep-0",
                                  "cpu-sleep-0", "WFI"])

        states = em.guess_idle_states([0, 1, 0, 0])
        self.assertEqual(states, ["cpu-sleep-0", "WFI",
                                  "cluster-sleep-0", "cluster-sleep-0",])

    def test_all_cpus_used(self):
        states = em.guess_idle_states([1, 1, 1, 1])
        self.assertEqual(states, ["WFI"] * 4)

    def test_one_cpu_per_cluster(self):
        states = em.guess_idle_states([0, 1, 0, 1])
        self.assertEqual(states, ["cpu-sleep-0", "WFI"] * 2)

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
