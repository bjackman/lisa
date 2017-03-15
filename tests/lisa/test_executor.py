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
from copy import deepcopy
import json
import shutil
import os
from unittest import TestCase

from mock import patch, Mock, MagicMock, call

import devlib

from env import TestEnv
from executor import Executor
import wlgen

class SetUpTarget(TestCase):
    def setUp(self):
        self.res_dir='test_{}'.format(self.__class__.__name__)
        self.te = TestEnv(target_conf={
            'platform': 'host',
            # TODO: can't calibrate rt-app on local targets because they're
            # probably intel_pstate and/or no root. Calibration requires setting
            # performance governor.
            'rtapp-calib': {i: 100 for i in range(4)},
            'modules': ['cgroups'],
        },
        test_conf={
            'results_dir': self.res_dir
        }, force_new=True)

mock_freezer = namedtuple('MockController', ['name'])('freezer')
class MockCgroupsModule(devlib.module.Module):
    name = 'cgroups'
    list_subsystems = Mock(return_value=[mock_freezer])
    freeze = Mock(name='Cgroups_freeze')
    @staticmethod
    def probe(target):
        return True

devlib.module.register_module(MockCgroupsModule)

example_wl = {
    "type" : "rt-app",
    "conf" : {
        "class" : "profile",
        "params" : {
            "mytask" : {
                "kind" : "Periodic",
                "params" : {
                    "duty_cycle_pct": 10,
                    "duration_s": 0.2,
                },
            },
        },
    }
}

class TestTaskNames(SetUpTarget):
    """Tests for the names of workload tasks created by the Executor"""

    def run_and_assert_task_names(self, experiments_conf, expected_tasks):
        executor = Executor(self.te, experiments_conf)
        executor.run()
        [experiment] = executor.experiments
        tasks = experiment.wload.tasks.keys()
        self.assertSetEqual(set(expected_tasks), set(tasks))

    def test_single_task_noprefix(self):
        experiments_conf = {
            'confs': [{
                'tag': 'myconf'
            }],
            "wloads" : {
                'mywl': example_wl
            },
        }

        self.run_and_assert_task_names(experiments_conf, ['task_mytask'])

    def test_single_task_prefix(self):
        wlspec = deepcopy(example_wl)

        wlspec['conf']['prefix'] = 'PREFIX'

        print wlspec

        experiments_conf = {
            'confs': [{
                'tag': 'myconf'
            }],
            "wloads" : {
                'mywl': wlspec
            },
        }

        self.run_and_assert_task_names(experiments_conf, ['PREFIXmytask'])

    def test_multiple_task(self):
        wlspec = deepcopy(example_wl)
        num_tasks = 5

        wlspec['conf']['params']['mytask']['tasks'] = num_tasks

        print wlspec

        experiments_conf = {
            'confs': [{
                'tag': 'myconf'
            }],
            "wloads" : {
                'mywl': wlspec
            },
        }

        exp_names = ['task_mytask_{}'.format(i) for i in range(num_tasks)]
        self.run_and_assert_task_names(experiments_conf, exp_names)

class TestWorkloadDuration(SetUpTarget):
    """Test that the 'duration' field for wlspecs is used"""
    def test_duration_profile(self):
        wlspec = deepcopy(example_wl)

        DURATION = 3

        wlspec['duration'] = DURATION

        print wlspec

        experiments_conf = {
            'confs': [{
                'tag': 'myconf'
            }],
            "wloads" : {
                'mywl': wlspec
            },
        }

        executor = Executor(self.te, experiments_conf)
        executor.run()
        [experiment] = executor.experiments
        with open(experiment.wload.json) as f:
            rtapp_conf = json.load(f)

        dur = rtapp_conf['global']['duration']
        self.assertEqual(dur, DURATION,
                         'Wrong global.duration field in rtapp JSON. '
                         'Expected {}, found {}'.format(DURATION, dur))

class TestMagicSmoke(SetUpTarget):
    def get_rta_results_dir(self, conf_name, wl_name):
        return os.path.join(self.te.LISA_HOME, 'results', self.res_dir,
                            'rtapp:{}:{}'.format(conf_name, wl_name))

    def test_files_created(self):
        """Test that we can run experiments and get output files"""
        conf_name = 'myconf'
        wl_name = 'mywl'

        results_dir = self.get_rta_results_dir(conf_name, wl_name)
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)

        experiments_conf = {
            'confs': [{
                'tag': conf_name
            }],
            "wloads" : {
                wl_name : example_wl,
            },
        }

        executor = Executor(self.te, experiments_conf)
        executor.run()

        self.assertTrue(
            os.path.isdir(results_dir),
            'Expected to find a directory at {}'.format(results_dir))

        result_1_dir = os.path.join(results_dir, '1')
        self.assertTrue(
            os.path.isdir(result_1_dir),
            'Expected to find a directory at {}'.format(result_1_dir))

class InterceptedRTA(wlgen.RTA):
    pre_callback = None
    def run(self, *args, **kwargs):
        self.pre_callback()
        super(InterceptedRTA, self).run(*args, **kwargs)

class BrokenRTAException(Exception):
    pass

class BrokenRTA(wlgen.RTA):
    pre_callback = None
    def run(self, *args, **kwargs):
        self.pre_callback()
        self._log.warning('\n\nInjecting workload failure\n')
        raise BrokenRTAException('INJECTED WORKLOAD FAILURE')

class TestFreezeUserspace(SetUpTarget):
    def _do_freezer_test(self):
        experiments_conf = {
            'confs': [{
                'tag': 'with_freeze',
                'flags': ['freeze_userspace'],
            }],
            "wloads" : {
                'my_wl' : example_wl,
            },
        }

        freezer_mock = self.te.target.cgroups.freeze
        freezer_mock.reset_mock()

        def assert_frozen(rta):
            freezer_mock.assert_called_once_with(
                ['init', 'systemd', 'sh', 'ssh'])
            freezer_mock.reset_mock()

        print wlgen.RTA
        wlgen.RTA.pre_callback = assert_frozen

        executor = Executor(self.te, experiments_conf)
        executor.run()

        freezer_mock.assert_called_once_with(thaw=True)

    @patch('wlgen.RTA', InterceptedRTA)
    def test_freeze_userspace(self):
        self._do_freezer_test()

    @patch('wlgen.RTA', BrokenRTA)
    def test_freeze_userspace_broken(self):
        self._do_freezer_test()
