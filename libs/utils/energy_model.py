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
from collections import namedtuple
# from trappy.stats.Topology import Topology

from devlib import TargetError

ActiveState = namedtuple("ActiveState", ["frequency", "capacity", "energy"])
IdleState = namedtuple("IdleState", ["energy"])

EnergyModelNode = namedtuple("EnergyModelNode",
                             ["cpus", "active_states", "idle_states"])

class EnergyModel(object):
    capacity_scale = 1024

    def __init__(self, topology, target):
        self.topology = topology

        self._levels = self._levels_from_target(target)

    def __repr__(self):
        repr_str = "EnergyModel:\n"
        for cluster_node in self._levels["cluster"]:
            cpus = cluster_node.cpus
            cpu_node = [n for n in self._levels["cpu"] if n.cpus[0] in cpus][0]

            repr_str += "cluster: {}\n".format(cpus)
            repr_str += "\t\tcpu" + 55*" " + "cluster\n"

            # Display active states from most to least power
            active_states = reversed(zip(cpu_node.active_states,
                                         cluster_node.active_states))

            for cpu_state, cluster_state in active_states:
                repr_str += "\t\t{:<30}\t{:>30}\n".format(cpu_state, cluster_state)

            for cpu_state, cluster_state in zip(cpu_node.idle_states,
                                                cluster_node.idle_states):
                repr_str += "\t\t{:<30}\t{:>30}\n".format(cpu_state, cluster_state)

        return repr_str

    def __iter__(self):
        return self._levels.__iter__()

    def _levels_from_target(self, target):
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

        # Assume a CPU always occurs in its own group 0
        nrg_dir_fmt = "{}/cpu{{}}/domain{{}}/group0/energy/".format(
            sched_domain_path)

        def parse_cap_states(cpu, sched_domain):
            path = nrg_dir_fmt.format(cpu, sched_domain) + "cap_states"
            vals = [int(v) for v in target.read_value(path).split()]
            # cap_states file is a list of (capacity, power) pairs
            return [(c, p) for c, p in zip(vals[::2], vals[1::2])]

        def parse_active_states(cpu, sched_domain):
            cap_states = parse_cap_states(cpu, sched_domain)

            cpufreq_freqs = target.cpufreq.list_frequencies(cpu)

            # TODO: we would like to have an expression of frequency domains as
            # a separate concept from "clusters". For now, just check they are
            # the same.
            assert target.cpufreq.get_domain_cpus(cpu) == cpus

            assert len(cpufreq_freqs) == len(cap_states)

            return [ActiveState(f, c, e) for f, (c, e) in
                    zip(cpufreq_freqs, cap_states)]

        def parse_idle_states(cpu, domain):
            path = nrg_dir_fmt.format(cpu, domain) + "idle_states"
            return [IdleState(int(v)) for v in target.read_value(path).split()]

        for cpus in topology.get_level("cluster"):
            cpu = cpus[0]

            #
            # Read CPU level data for this cluster
            #

            active_states = parse_active_states(cpu, 0)
            idle_states = parse_idle_states(cpu, 0)

            for cpu in cpus:
                node = EnergyModelNode([cpu], active_states, idle_states)
                levels["cpu"].append(node)

            #
            # Read cluster-level data
            #

            cap_states = parse_cap_states(cpu, 1)
            idle_states = parse_idle_states(cpu, 0)

            node = EnergyModelNode(cpus, active_states, idle_states)
            levels["cluster"].append(node)

        return levels

        # for level_idx, level_name in enumerate(topology):
        #     topo_level = topology.get_level(level_name)


    def get_level(self, level):
        return self._levels[level]
