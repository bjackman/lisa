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
import unittest

from bart.common.Utils import select_window
from devlib.target import TargetError
from devlib.utils.misc import memoized
from trappy.stats.grammar import Parser

from env import TestEnv
from test import LisaTest, experiment_test
from trace import Trace

class PeltSmokeTest(LisaTest):
    """
    Pelt Smoke Test

    Run workloads providing a known utilization and test that the PELT
    (Per-entity load tracking) util_avg signal reflects that utilization.

    This does not test capacity or frequency invariance - the workload is run on
    a "biggest" CPU at the maximum frequency.
    """

    # The time, in seconds, we allow for the signal to "settle"
    util_avg_convergence_time = 0.15

    test_conf = {
        "tools"    : [ "rt-app" ],
        "ftrace" : {
            "events" : [
                "sched_switch",
                "sched_load_avg_task",
                "sched_load_avg_cpu",
            ],
        },
    }

    experiments_conf = {
        "wloads": {}, # Created in _getTestConf
        "confs" : [
            {
                "tag" : "",
                "flags" : "ftrace",
                "cpufreq" : {
                    "governor" : "performance"
                },
            }
        ],
    }

    # The common parts of the rt-app workload configuration. Target-dependent
    # factors will be set up in _getExperimentsConf.
    wload_base = {
        "type" : "rt-app",
        "conf" : {
            "class" : "periodic",
            "params" : {
                # "duty_cycle_pct": set in _getTestConf
                "duration_s": 5,
                "period_ms": 3,
            },
            # "cpus" : set in _getTestConf
            "tasks" : 1,
            "prefix" : "lt_test",
        },
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(PeltSmokeTest, cls)._init(*args, **kwargs)

    @memoized
    @classmethod
    def _get_cpu(cls, target):
        # Run on a "big" CPU, or any CPU if not big.LITTLE
        if hasattr(target, "bl"):
            return target.bl.bigs[0]
        else:
            return 0

    @classmethod
    def _getExperimentsConf(cls, test_env):
        for duty_cycle in [30, 50, 70]:
            wload = dict(cls.wload_base)
            wload["conf"]["params"]["duty_cycle_pct"] = duty_cycle
            wload["conf"]["cpus"] = [cls._get_cpu(test_env.target)]

            name = "wl_{}pct".format(duty_cycle)
            cls.experiments_conf["wloads"][name] = wload

        return cls.experiments_conf

    @experiment_test
    def test_task_util(self, experiment, tasks):
        # Examine workload to figure out expected util_avg value
        [task] = tasks
        params = experiment.wload.params["profile"]
        [phase] = params[task]["phases"]
        logging.info("Testing {}% task".format(phase.duty_cycle_pct))
        exp_util = (phase.duty_cycle_pct * 1024) / 100.

        events = self.test_conf["ftrace"]["events"]
        trace = Trace(self.te.platform, experiment.out_dir, events)

        # Get time during which workload ran
        (wload_start, wload_end) = self.get_window(experiment)
        # Ignore an initial period for the signal to settle
        window = (wload_start + self.util_avg_convergence_time, wload_end)

        [pid] = trace.getTaskByName(task)
        parser = Parser(trace.ftrace, filters={"pid": pid})
        util_avg_all = parser.solve("sched_load_avg_task:util_avg")[pid]
        util_avg = select_window(util_avg_all, self.get_window(experiment))

        error_margin = exp_util * 0.03
        self.assertAlmostEqual(util_avg.mean(), exp_util, delta=error_margin)
