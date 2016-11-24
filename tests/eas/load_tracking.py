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

from bart.common.Utils import select_window, area_under_curve
from devlib.utils.misc import memoized
from trappy.stats.grammar import Parser

from test import LisaTest, experiment_test
from trace import Trace

UTIL_SCALE = 1024
# Time in seconds to allow for util_avg to converge (i.e. ignored time)
UTIL_AVG_CONVERGENCE_TIME = 0.15
# Allowed margin between expected and observed util_avg value
ERROR_MARGIN_PCT = 15

class FreqInvarianceTest(LisaTest):
    """
    Frequency invariance test for util_avg signal

    Run workloads providing a known utilization and test that the PELT
    (Per-entity load tracking) util_avg signal reflects that utilization at
    various CPU frequencies.
    """

    test_conf = {
        "tools"    : [ "rt-app" ],
        "ftrace" : {
            "events" : [
                "sched_switch",
                "sched_load_avg_task",
                "sched_load_avg_cpu",
                "sched_pelt_se",
            ],
        },
        "modules": ["cpufreq"],
    }

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        super(FreqInvarianceTest, cls)._init(*args, **kwargs)

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
        cpu = cls._get_cpu(test_env.target)

        # 10% periodic RTApp workload:
        wloads = {
            "fie_10pct" : {
                "type" : "rt-app",
                "conf" : {
                    "class" : "periodic",
                    "params" : {
                        "duty_cycle_pct": 10,
                        "duration_s": 5,
                        "period_ms": 10,
                    },
                    "tasks" : 1,
                    "prefix" : "fie_test",
                    "cpus" : [cpu]
                },
            },
        }

        confs = []

        # Create a set of confs with different frequencies
        # We'll run the 10% workload under each conf (i.e. at each frequency)

        all_freqs = test_env.target.cpufreq.list_frequencies(cpu)
        # If we have loads of frequencies just test a subset
        freqs = all_freqs[::len(all_freqs)/8 + 1]
        # Don't test the devil's frequency
        freqs = [f for f in freqs if f != 666666666]
        for freq in freqs:
            confs.append({
                "tag" : "freq_{}".format(freq),
                "flags" : "ftrace",
                "cpufreq" : {
                    "freqs" : {cpu: freq},
                    "governor" : "userspace",
                },
            })

        return {
            "wloads": wloads,
            "confs": confs,
        }

    @experiment_test
    def test_task_util(self, experiment, tasks):
        """
        Assert that the mean of the util_avg signal matched the expected value
        """
        # Examine workload to figure out expected util_avg value
        [task] = tasks
        params = experiment.wload.params["profile"]
        [phase] = params[task]["phases"]
        logging.info("Testing {}% task".format(phase.duty_cycle_pct))
        exp_util = (phase.duty_cycle_pct * UTIL_SCALE) / 100.

        # Get trace
        events = self.test_conf["ftrace"]["events"]
        trace = Trace(self.te.platform, experiment.out_dir, events, tasks)

        # The Parser below will error out very cryptically if there are none of
        # the required events in the trace - catch it here instead.
        if "sched_load_avg_task" in trace.available_events:
            event = "sched_load_avg_task"
        elif "sched_pelt_se" in trace.available_events:
            event = "sched_pelt_se"
        else:
            raise unittest.SkipTest(
                "No sched_load_avg_task or sched_pelt_se events. "
                "Does the kernel support them?")

        # Get time window during which workload ran
        (wload_start, wload_end) = self.get_window(experiment)
        # Ignore an initial period for the signal to settle
        window = (wload_start + UTIL_AVG_CONVERGENCE_TIME, wload_end)

        # Find mean value for util_avg
        [pid] = trace.getTaskByName(task)
        # TODO: should do the following:
        # parser = Parser(trace.ftrace, filters={"pid": pid})
        # util_avg_all = parser.solve("{}:util_avg".format(event))[pid]
        # But something's wrong with Trappy (deep errors from pyparsing)
        # Instead just mangle DataFrames by hand:
        df = trace.ftrace.sched_pelt_se.data_frame
        util_avg_all = df[df["__comm"].isin(tasks)]["util_avg"]
        util_avg = select_window(util_avg_all, window)
        util_avg_mean = area_under_curve(util_avg) / (window[1] - window[0])

        error_margin = exp_util * (ERROR_MARGIN_PCT / 100.)
        freq = experiment.conf["cpufreq"]["freqs"].values()[0]
        msg = "Saw util_avg around {}, expected {} at freq {}".format(
            util_avg_mean, exp_util, freq)
        self.assertAlmostEqual(util_avg_mean, exp_util, delta=error_margin,
                               msg=msg)
