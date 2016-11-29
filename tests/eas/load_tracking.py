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

    def get_expected_util_avg(self, experiment):
        """
        Examine workload to figure out expected util_avg value

        Assumes an RT-App workload with a single task with a single phase.
        """
        [task] = experiment.wload.tasks.keys()
        params = experiment.wload.params["profile"]
        [phase] = params[task]["phases"]
        return (phase.duty_cycle_pct * UTIL_SCALE) / 100.

    def get_sched_signal(self, experiment, signal):
        """
        Get a pandas.Series with the sched signal for the workload task

        This examines scheduler load tracking trace events, supporting either
        sched_load_avg_task or sched_pelt_se. You will need a target kernel that
        includes these events.
        """
        [task] = experiment.wload.tasks.keys()

        events = self.test_conf["ftrace"]["events"]
        trace = Trace(self.te.platform, experiment.out_dir, events, [task])

        # There are two different scheduler trace events that expose the
        # util_avg signal. Neither of them is in mainline. Eventually they
        # should be unified but for now we'll just check for both types of
        # event.
        if "sched_load_avg_task" in trace.available_events:
            event = "sched_load_avg_task"
        elif "sched_pelt_se" in trace.available_events:
            event = "sched_pelt_se"
        else:
            raise ValueError("No sched_load_avg_task or sched_pelt_se events. "
                             "Does the kernel support them?")

        df = getattr(trace.ftrace, event).data_frame
        util_avg = df[df["__comm"].isin([task])][signal]
        return select_window(util_avg, self.get_window(experiment))

    def get_signal_mean(self, experiment, signal,
                        ignore_first_s=UTIL_AVG_CONVERGENCE_TIME):
        """
        Get the mean of a scheduler signal for the experiment's task

        Ignore the first `ignore_first_s` seconds of the signal.
        """
        (wload_start, wload_end) = self.get_window(experiment)
        window = (wload_start + ignore_first_s, wload_end)

        util_avg = self.get_sched_signal(experiment, signal)
        return area_under_curve(util_avg) / (window[1] - window[0])

    @experiment_test
    def test_task_util(self, experiment, tasks):
        """
        Test that the mean of the util_avg signal matched the expected value
        """
        exp_util = self.get_expected_util_avg(experiment)
        util_avg_mean = self.get_signal_mean(experiment, "util_avg")

        error_margin = exp_util * (ERROR_MARGIN_PCT / 100.)
        freq = experiment.conf["cpufreq"]["freqs"].values()[0]
        msg = "Saw util_avg around {}, expected {} at freq {}".format(
            util_avg_mean, exp_util, freq)
        self.assertAlmostEqual(util_avg_mean, exp_util, delta=error_margin,
                               msg=msg)

    @experiment_test
    def test_task_load(self, experiment, tasks):
        """
        Test that the mean of the load_avg signal matched the expected value
        """
        # Assuming that the system was under little stress (so the task was
        # RUNNING whenever it was RUNNABLE) and that the task was run with a
        # 'nice' value of 0, the load_avg should be similar to the util_avg.
        exp_load = self.get_expected_util_avg(experiment)
        load_avg_mean = self.get_signal_mean(experiment, "load_avg")

        error_margin = exp_load * (ERROR_MARGIN_PCT / 100.)
        freq = experiment.conf["cpufreq"]["freqs"].values()[0]
        msg = "Saw load_avg around {}, expected {} at freq {}".format(
            load_avg_mean, exp_load, freq)
        self.assertAlmostEqual(load_avg_mean, exp_load, delta=error_margin,
                               msg=msg)
