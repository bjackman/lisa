# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017, ARM Limited and contributors.
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

import json
import os
from unittest import TestCase

from trace import Trace

class TestTrace(TestCase):
    """Smoke tests for LISA's Trace class"""

    traces_dir = os.path.join(os.path.dirname(__file__),
                              'example_traces')
    events = ['cpu_frequency', 'cpu_idle',
              'sched_migrate_task', 'sched_switch']

    def __init__(self, *args, **kwargs):
        super(TestTrace, self).__init__(*args, **kwargs)

        with open(os.path.join(self.traces_dir, 'platform.json')) as f:
            platform = json.load(f)

        trace_path = os.path.join(self.traces_dir, 'trace.dat')
        self.trace = Trace(platform, trace_path, self.events)

    def test_getTaskByName(self):
        for name, pids in [('task_wmig_0', [2453]),
                           ('true', [2440, 2442, 2444, 2446, 2448, 2450]),
                           ('NOT_A_TASK', [])]:
            self.assertEqual(self.trace.getTaskByName(name), pids)

    def test_getTaskByPid(self):
        for pid, names in [(1, 'systemd'),
                           (2453, 'task_wmig_0'),
                           (123456, None)]:
            self.assertEqual(self.trace.getTaskByPid(pid), names)

    def test_getTasks(self):
        tasks_dict = self.trace.getTasks()
        print tasks_dict.keys()
        for pid, name in [(1, 'systemd'),
                          (2453, 'task_wmig_0'),
                          (2442, 'true'),
                          (2444, 'true')]:
            self.assertEqual(tasks_dict[pid], name)



