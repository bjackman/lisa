from energy_model import ActiveState, EnergyModelNode, PowerDomain, EnergyModel
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
from energy_model import (ActiveState, EnergyModelNode, EnergyModelRoot,
                          PowerDomain, EnergyModel)

from collections import OrderedDict

cluster_active_states = OrderedDict([
    (208000, ActiveState(capacity=178, power=16)),
    (432000, ActiveState(capacity=369, power=29)),
    (729000, ActiveState(capacity=622, power=47)),
    (960000, ActiveState(capacity=819, power=75)),
    (1200000, ActiveState(capacity=1024, power=112))
])

cluster_idle_states = OrderedDict([
    ('WFI', 107),
    ('cpu-sleep', 47),
    ('cluster-sleep', 0)
])

cpu_active_states = OrderedDict([
    (208000,  ActiveState(capacity=178,  power=69)),
    (432000,  ActiveState(capacity=369,  power=125)),
    (729000,  ActiveState(capacity=622,  power=224)),
    (960000,  ActiveState(capacity=819,  power=367)),
    (1200000, ActiveState(capacity=1024, power=670))
])

cpu_idle_states = OrderedDict([('WFI', 15), ('cpu-sleep', 0), ('cluster-sleep', 0)])

cpus = range(8)

cluster_pds = [
    PowerDomain(cpus=[0, 1, 2, 3], idle_states=["cluster-sleep"], parent=None),
    PowerDomain(cpus=[4, 5, 6, 7], idle_states=["cluster-sleep"], parent=None),
]

def cpu_node(cpu):
    cpu_pd=PowerDomain(cpus=[cpu],
                       parent=cluster_pds[cpu / 4],
                       idle_states=["WFI", "cpu-sleep"])

    return EnergyModelNode([cpu],
                           active_states=cpu_active_states,
                           idle_states=cpu_idle_states,
                           power_domain=cpu_pd,
                           freq_domain=cpus)
hikey_energy_levels = [
    [cpu_node(c) for c in cpus],
    [
        EnergyModelNode(cpus=[0, 1, 2, 3],
                        active_states=cluster_active_states,
                        idle_states=cluster_idle_states),
        EnergyModelNode(cpus=[4, 5, 6, 7],
                        active_states=cluster_active_states,
                        idle_states=cluster_idle_states)
    ],
]

hikey_energy = EnergyModel(levels=hikey_energy_levels)
