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

from bisect import bisect
from math import isnan

import numpy as np
import pandas as pd

from bart.common.Utils import area_under_curve, select_window
from energy_model import EnergyModelCapacityError
from perf_analysis import PerfAnalysis
from test import experiment_test
from trace import Trace
from . import (EasTest, energy_aware_conf,
               WORKLOAD_DURATION_S,
               WORKLOAD_PERIOD_MS)

class EnergyModelTest(EasTest):
    """
    Goal
    ====

    The arrangement of an arbitrary set of tasks is the most energy efficient.

    Detailed Description
    ====================

    Take a single stage workload and search for the most energy-efficient task
    placements for that workload.

    Expected Behaviour
    ==================

    The tasks were placed according to one of the optimal placements.
    """

    negative_slack_allowed_pct = 5
    """Percentage of RT-App task activations with negative slack allowed"""

    @classmethod
    def _getExperimentsConf(cls, *args, **kwargs):
        return {
            'wloads' : cls.workloads,
            'confs' : [energy_aware_conf]
        }

    def get_task_utils_df(self, experiment):
        """
        Get a DataFrame with the expected utilization of each task over time
        """
        util_scale = self.te.nrg_model.capacity_scale

        transitions = {}
        def add_transition(time, task, util):
            if time not in transitions:
                transitions[time] = {task: util}
            else:
                transitions[time][task] = util

        # First we'll build a dict D {time: {task_name: util}} where D[t][n] is
        # the expected utilization of task n from time t.
        for task, params in experiment.wload.params['profile'].iteritems():
            time = self.get_start_time(experiment) + params['delay']
            add_transition(time, task, 0)
            for _ in range(params.get('loops', 1)):
                for phase in params['phases']:
                    util = (phase.duty_cycle_pct * util_scale / 100.)
                    add_transition(time, task, util)
                    time += phase.duration_s
            add_transition(time, task, 0)

        index = sorted(transitions.keys())
        df = pd.DataFrame([transitions[k] for k in index], index=index)
        return df.fillna(method='ffill')

    def get_task_cpu_df(self, experiment):
        tasks = experiment.wload.tasks.keys()
        trace = self.get_trace(experiment)

        df = trace.ftrace.sched_switch.data_frame[['next_comm', '__cpu']]
        df = df[df['next_comm'].isin(tasks)]
        df = df.pivot(index=df.index, columns='next_comm').fillna(method='ffill')
        cpu_df = df['__cpu']
        # Drop consecutive duplicates
        cpu_df = cpu_df[(cpu_df.shift(+1) != cpu_df).any(axis=1)]
        return cpu_df

    def get_power_df(self, experiment, task_cpu_df=None, task_utils_df=None):
        if task_cpu_df is None:
            task_cpu_df = self.get_task_cpu_df(experiment)
        if task_utils_df is None:
            task_utils_df = self.get_task_utils_df(experiment)

        tasks = experiment.wload.tasks.keys()

        # Create a combined DataFrame with the utilization of a task and the CPU
        # it was running on at each moment. Looks like:
        #                       utils                  cpus
        #          task_wmig0 task_wmig1 task_wmig0 task_wmig1
        # 2.375056      102.4      102.4        NaN        NaN
        # 2.375105      102.4      102.4        2.0        NaN

        df = pd.concat([task_utils_df, task_cpu_df],
                       axis=1, keys=['utils', 'cpus'])
        df = df.sort_index().fillna(method='ffill')
        nrg_model = self.executor.te.nrg_model

        # Now make a DataFrame with the estimated power at each moment.
        def est_power(row):
            cpu_utils = [0 for cpu in nrg_model.cpus]
            for task in tasks:
                cpu = row['cpus'][task]
                util = row['utils'][task]
                if not isnan(cpu):
                    cpu_utils[int(cpu)] += util
            power = nrg_model.estimate_from_cpu_util(cpu_utils)
            columns = power.keys()
            return pd.Series([power[c] for c in columns], index=columns)
        return df.apply(est_power, axis=1)

    def get_expected_power_df(self, experiment, task_utils_df=None):
        if task_utils_df is None:
            task_utils_df = self.get_task_utils_df(experiment)

        nrg_model = self.te.nrg_model

        def exp_power(row):
            task_utils = row.to_dict()
            expected_utils = nrg_model.get_optimal_placements(task_utils)
            power = nrg_model.estimate_from_cpu_util(expected_utils[0])
            columns = power.keys()
            return pd.Series([power[c] for c in columns], index=columns)
        return task_utils_df.apply(exp_power, axis=1)

    @experiment_test
    def test_slack(self, experiment, tasks):
        """Test that the RTApp workload was given enough performance"""

        pa = PerfAnalysis(experiment.out_dir)
        for task in tasks:
            slack = pa.df(task)["Slack"]

            bad_activations_pct = len(slack[slack < 0]) * 100. / len(slack)
            if bad_activations_pct > self.negative_slack_allowed_pct:
                raise AssertionError("task {} missed {}% of activations".format(
                    task, bad_activations_pct))

    @experiment_test
    def test_task_placement(self, experiment, tasks):
        exp_power = self.get_expected_power_df(experiment)
        est_power = self.get_power_df(experiment)

        exp_energy = area_under_curve(exp_power['power'])
        est_energy = area_under_curve(est_power['power'])

        msg = 'Estimated {} bogo-Joules to run workload, expected {}'.format(
            est_energy, exp_energy)
        self.assertLess(est_energy, exp_energy * 1.2, msg=msg)

class OneSmallTask(EnergyModelTest):
    workloads = {
        'one_small' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 20,
                    'duration_s': 5,
                    'period_ms': 10,
                },
                'tasks' : 1,
                'prefix' : 'many',
            },
        }
    }

class ThreeSmallTasks(EnergyModelTest):
    workloads = {
        'three_small' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 20,
                    'duration_s': 5,
                    'period_ms': 10,
                },
                'tasks' : 3,
                'prefix' : 'many',
            },
        }
    }

class TwoBigTasks(EnergyModelTest):
    workloads = {
        'two_big' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'periodic',
                'params' : {
                    'duty_cycle_pct': 90,
                    'duration_s': 5,
                    'period_ms': 10,
                },
                'tasks' : 2,
                'prefix' : 'many',
            },
        }
    }

class TwoBigThreeSmall(EnergyModelTest):
    workloads = {
        'mixed' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'profile',
                'params' : {
                    'large' : {
                        'kind' : 'Periodic',
                        'params' : {
                            'duty_cycle_pct': 70,
                            'duration_s': WORKLOAD_DURATION_S,
                            'period_ms': WORKLOAD_PERIOD_MS,
                        },
                        'tasks' : 2,
                    },
                    'small' : {
                        'kind' : 'Periodic',
                        'params' : {
                            'duty_cycle_pct': 10,
                            'duration_s': WORKLOAD_DURATION_S,
                            'period_ms': WORKLOAD_PERIOD_MS,
                        },
                        'tasks' : 3,
                    },
                },
            },
        },
    }

class EnergyModelWakeMigration(EnergyModelTest):
    workloads = {
        'wake_migration' : {
            'type' : 'rt-app',
            'conf' : {
                'class' : 'profile',
                'params' : {
                    'wmig' : {
                        'kind' : 'Step',
                        'params' : {
                            'start_pct': 10,
                            'end_pct': 50,
                            'time_s': 1,
                            'loops': 2
                        },
                        # Create one task for each big cpu
                        'tasks' : 'big',
                    },
                },
            },
        },
    }
