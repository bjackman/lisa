# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
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
import json
import operator
import os
import trappy
import unittest

from bart.sched.SchedAssert import SchedAssert
from bart.sched.SchedMultiAssert import SchedMultiAssert
from devlib.target import TargetError

from env import TestEnv
from devlib.utils.misc import memoized
from test import LisaTest, experiment_test

logging.getLogger().setLevel(logging.DEBUG)

# Read the config file and update the globals
CONF_FILE = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    "acceptance.config")

with open(CONF_FILE, "r") as fh:
    conf_vars = json.load(fh)
    globals().update(conf_vars)

class EasTest(LisaTest):
    """
    Base class for EAS tests
    """

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        conf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 cls.conf_basename)

        super(EasTest, cls)._init(conf_file, *args, **kwargs)

    @memoized
    def get_multi_assert(self, experiment, task_filter=""):
        tasks = experiment.wload.tasks.keys()
        return SchedMultiAssert(experiment.out_dir,
                                self.te.topology,
                                [t for t in tasks if task_filter in t])

    @classmethod
    def _experimentsInit(cls, *args, **kwargs):
        super(EasTest, cls)._experimentsInit(*args, **kwargs)

        if SET_INITIAL_TASK_UTIL:
            cls.target.write_value(
                "/proc/sys/kernel/sched_initial_task_util", 1024)

        if SET_IS_BIG_LITTLE:
            try:
                cls.target.write_value(
                    "/proc/sys/kernel/sched_is_big_little", 1)
            except TargetError:
                # That flag doesn't exist on mainline-integration kernels, so
                # don't worry if the file isn't present.
                pass

    def get_start_time(self, experiment):
        start_times_dict = self.get_multi_assert(experiment).getStartTime()
        return min([t["starttime"] for t in start_times_dict.itervalues()])

    def _do_test_first_cpu(self, experiment, tasks):
        """Test that all tasks start on a big CPU"""

        sched_assert = self.get_multi_assert(experiment)

        self.assertTrue(
            sched_assert.assertFirstCpu(
                self.target.bl.bigs,
                rank=len(tasks)),
            msg="Not all the new generated tasks started on a big CPU")

class SingleTaskLowestEnergy(EasTest):
    """
    Goal
    ====

    Check that a lone task in the system is placed according to the lowest
    energy cost.

    Detailed Description
    ====================

    Run workloads that consist of a single task, assuming the other load on the
    system is negligible, and that no other configuration, such as boosting with
    schedtune, has been done.


    Expected Behaviour
    ==================

    Ths single task should be placed on the CPU/OPP combination that uses the
    least energy without exceeding a certain utilization level on the CPU.

    """

    conf_basename = "single_task.config"

    @experiment_test
    def test_least_energy(self, experiment, tasks):
        assert len(tasks) == 1
        task = tasks[0]

        # TODO: I don't think the topological level should be explicit, but
        # that's the way trappy works. Any way around this?
        topo_level = "cluster"

        # TODO: determine
        # this from the workload, so we can run multiple workloads and re-use
        # the test
        task_util = 0.2
        # TODO configure 20% margin
        required_capacity = task_util * 1.2 * self.te.nrg_model.capacity_scale

        # Find CPU/OPP pairs that could contain the task
        candidates = []
        for cpu_nrg in self.te.nrg_model.get_level(topo_level):
            possible_states = [s for s in cpu_nrg.active_states
                               if s.capacity > required_capacity]
            if not possible_states:
                # This CPU can't handle this task at any frequency
                continue

            best_state = min(possible_states, key=lambda s: s.energy)
            best_state_idx = cpu_nrg.active_states.index(best_state)
            candidates.append((cpu_nrg, best_state_idx))
            logging.debug(
                "Could run capacity {} on CPUs {} at OPP {} or above".format(
                    required_capacity, cpu_nrg.cpus, best_state_idx))

        assert len(candidates)

        # Find which capacity group would use the least energy while running the
        # task

        # TODO: Would this be better as a method of the energy model that takes
        # a distribution of utilisation and guesses energy based on an educated
        # guess about idle states?
        candidates_nrg = []
        for cpu_nrg, cap_idx in candidates:
            # Within the capacity group it doesn't matter which CPU ran the task
            cpu = cpu_nrg.cpus[0]

            energy = 0
            for level in self.te.nrg_model:
                for node in self.te.nrg_model.get_level(level):
                    if cpu in node.cpus:
                        util = 0.2
                    else:
                        util = 0

                    # Assume only the shallowest idle state is used
                    # TODO come up with a way of modeling idle states
                    idle_energy = (1 - util) * node.idle_states[0].energy
                    active_energy = util * node.active_states[cap_idx].energy

                    energy += idle_energy + active_energy

            logging.debug("Could CPUs {} at OPP {} would use energy {}".format(
                cpu_nrg.cpus, best_state_idx, energy))
            candidates_nrg.append((cpu_nrg, cap_idx, energy))

        best_nrg, best_cap_idx, _ = min(candidates_nrg, key=lambda c: c[2])

        # This doesn't actually need to be a SchedMultiAssert, could just be a
        # SchedAssert since we only have one task.
        sched_assert = self.get_multi_assert(experiment)
        self.assertTrue(
            sched_assert.assertResidency(
                topo_level,
                best_nrg.cpus,
                task_util * 0.9 * 100,
                operator.ge,
                percent=True,
                rank=len(tasks)),
            msg="Didn't run on expected cores")

class ForkMigration(EasTest):
    """
    Goal
    ====

    Check that newly created threads start on a big CPU

    Detailed Description
    ====================

    The test spawns as many threads as there are cores in the system.
    It then checks that all threads started on a big core.

    Expected Behaviour
    ==================

    The threads start on a big core.
    """

    conf_basename = "fork_migration.config"

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Fork Migration: Test First CPU"""
        self._do_test_first_cpu(experiment, tasks)

class SmallTaskPacking(EasTest):
    """
    Goal
    ====

    Many small tasks are packed in little cpus

    Detailed Description
    ====================

    The tests spawns as many tasks as there are cpus in the system.
    The tasks are small, so none of them should be run on big cpus and
    the scheduler should pack them on little cpus.

    Expected Behaviour
    ==================

    All tasks run on little cpus.
    """

    conf_basename = "small_task_packing.config"

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Small Task Packing: test first CPU"""
        self._do_test_first_cpu(experiment, tasks)

    @experiment_test
    def test_small_task_residency(self, experiment, tasks):
        "Small Task Packing: Test Residency (Little Cluster)"

        sched_assert = self.get_multi_assert(experiment)

        self.assertTrue(
            sched_assert.assertResidency(
                "cluster",
                self.target.bl.littles,
                EXPECTED_RESIDENCY_PCT,
                operator.ge,
                percent=True,
                rank=len(tasks)),
            msg="Not all tasks are running on LITTLE cores for at least {}% of their execution time"\
                    .format(EXPECTED_RESIDENCY_PCT))

class OffloadMigrationAndIdlePull(EasTest):
    """
    Goal
    ====

    Big cpus pull big tasks from little cpus when they become idle

    Detailed Description
    ====================

    This test runs twice as many tasks are there are big cpus.  All
    these tasks are big tasks.  Half of them are called
    "early_starter" and the other half "migrator".  The migrator tasks
    start 1 second after the early_starter tasks.  As the big cpus are
    fully utilized when the migrator tasks start, some tasks are
    offloaded to the little cpus.  As the big cpus finish their tasks,
    they pull tasks from the little to complete them.

    Expected Behaviour
    ==================

    As there are as many early_starter tasks as there are big cpus,
    the early_starter tasks should run in the big cpus until they
    finish.  When the migrator tasks start, there is no spare capacity
    in the big cpus so they run on the little cpus.  Once the big cpus
    finish with the early_starters, they should pull the migrator
    tasks and run them.

    It is possible that when the migrator tasks start they do it on
    big cpus and they end up displacing the early starters.  This is
    acceptable behaviour.  As long as big cpus are fully utilized
    running big tasks, the scheduler is doing a good job.

    That is why this test doesn't test for migrations of the migrator
    tasks to the bigs when we expect that the early starters have
    finished.  Instead, it tests that:

      * The big cpus were fully loaded as long as there are tasks left
        to run in the system

      * The little cpus run tasks while the bigs are busy (offload migration)

      * All tasks get a chance on a big cpu (either because they
        started there or because of idle pull)

      * All tasks are finished off in a big cpu.

    """

    conf_basename = "offload_idle_pull.config"

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Offload Migration and Idle Pull: Test First CPU"""
        self._do_test_first_cpu(experiment, tasks)

    @classmethod
    def calculate_end_times(cls, experiment):

        end_times = {}
        for task in experiment.wload.tasks.keys():
            sched_assert = SchedAssert(experiment.out_dir, cls.te.topology,
                                       execname=task)
            end_times[task] = sched_assert.getEndTime()

        return end_times

    @experiment_test
    def test_big_cpus_fully_loaded(self, experiment, tasks):
        """Offload Migration and Idle Pull: Big cpus are fully loaded as long as there are tasks left to run in the system"""
        num_big_cpus = len(self.target.bl.bigs)

        sched_assert = self.get_multi_assert(experiment)

        end_times = sorted(self.calculate_end_times(experiment).values())

        # Window of time until the first migrator finishes
        window = (self.get_start_time(experiment), end_times[-num_big_cpus])
        busy_time = sched_assert.getCPUBusyTime("cluster",
                                            self.target.bl.bigs,
                                            window=window, percent=True)

        msg = "Big cpus were not fully loaded while there were enough big tasks to fill them"
        self.assertGreater(busy_time, OFFLOAD_EXPECTED_BUSY_TIME_PCT, msg=msg)

        # As the migrators start finishing, make sure that the tasks
        # that are left are running on the big cpus
        for i in range(num_big_cpus-1):
            big_cpus_left = num_big_cpus - i - 1
            window = (end_times[-num_big_cpus+i], end_times[-num_big_cpus+i+1])
            busy_time = sched_assert.getCPUBusyTime("cluster",
                                                    self.target.bl.bigs,
                                                    window=window, percent=True)

            expected_busy_time = OFFLOAD_EXPECTED_BUSY_TIME_PCT * \
                                 big_cpus_left / num_big_cpus
            msg = "Big tasks were not running on big cpus from {} to {}".format(
                window[0], window[1])

            self.assertGreater(busy_time, expected_busy_time, msg=msg)

    @experiment_test
    def test_little_cpus_run_tasks(self, experiment, tasks):
        """Offload Migration and Idle Pull: Little cpus run tasks while bigs are busy"""

        num_offloaded_tasks = len(tasks) / 2

        end_times = self.calculate_end_times(experiment).values()
        first_task_finish_time = min(end_times)

        migrators_assert = self.get_multi_assert(experiment, "migrator")
        start_time = min(t["starttime"]
                         for t in migrators_assert.getStartTime().itervalues())
        migrator_activation_time = start_time + OFFLOAD_MIGRATION_MIGRATOR_DELAY

        window = (migrator_activation_time, first_task_finish_time)

        all_tasks_assert = self.get_multi_assert(experiment)

        busy_time = all_tasks_assert.getCPUBusyTime("cluster",
                                                    self.target.bl.littles,
                                                    window=window)

        window_len = window[1] - window[0]
        expected_busy_time = window_len * num_offloaded_tasks * \
                             OFFLOAD_EXPECTED_BUSY_TIME_PCT / 100.
        msg = "Little cpus did not pick up big tasks while big cpus were fully loaded"

        self.assertGreater(busy_time, expected_busy_time, msg=msg)

    @experiment_test
    def test_all_tasks_run_on_a_big_cpu(self, experiment, tasks):
        """Offload Migration and Idle Pull: All tasks run on a big cpu at some point

        Note: this test may fail in big.LITTLE platforms in which the
        little cpus are almost as performant as the big ones.

        """
        for task in tasks:
            sa = SchedAssert(experiment.out_dir, self.te.topology, execname=task)
            end_times = self.calculate_end_times(experiment)
            window = (0, end_times[task])
            big_residency = sa.getResidency("cluster", self.target.bl.bigs,
                                            window=window, percent=True)

            msg = "Task {} didn't run on a big cpu.".format(task)
            self.assertGreater(big_residency, 0, msg=msg)

    @experiment_test
    def test_all_tasks_finish_on_a_big_cpu(self, experiment, tasks):
        """Offload Migration and Idle Pull: All tasks finish on a big cpu

        Note: this test may fail in big.LITTLE systems where the
        little cpus' performance is comparable to the bigs' and they
        can take almost the same time as a big cpu to complete a
        task.

        """
        for task in tasks:
            sa = SchedAssert(experiment.out_dir, self.te.topology, execname=task)

            msg = "Task {} did not finish on a big cpu".format(task)
            self.assertIn(sa.getLastCpu(), self.target.bl.bigs, msg=msg)


class WakeMigration(EasTest):
    """
    Goal
    ====

    A task that switches between being high and low utilization moves
    to big and little cores accordingly

    Detailed Description
    ====================

    This test creates as many tasks as there are big cpus.  The tasks
    alternate between high and low utilization.  They start being
    small load for 5 seconds, they become big for another 5 seconds,
    then small for another 5 seconds and finally big for the last 5
    seconds.

    Expected Behaviour
    ==================

    The tasks should run on the litlle cpus when they are small and in
    the big cpus when they are big.
    """

    conf_basename = "wake_migration.config"

    @experiment_test
    def test_first_cpu(self, experiment, tasks):
        """Wake Migration: Test First CPU"""
        self._do_test_first_cpu(experiment, tasks)

    def _assert_switch(self, experiment, expected_switch_to, phases):
        if expected_switch_to == "big":
            switch_from = self.target.bl.littles
            switch_to   = self.target.bl.bigs
        elif expected_switch_to == "little":
            switch_from = self.target.bl.bigs
            switch_to   = self.target.bl.littles
        else:
            raise ValueError("Invalid expected_switch_to")

        sched_assert = self.get_multi_assert(experiment)

        expected_time = (self.get_start_time(experiment)
                         + phases*WORKLOAD_DURATION_S)
        switch_window = (max(expected_time - SWITCH_WINDOW_HALF, 0),
                         expected_time + SWITCH_WINDOW_HALF)

        fmt = "Not all tasks wake-migrated to {} cores in the expected window: {}"
        msg = fmt.format(expected_switch_to, switch_window)

        self.assertTrue(
            sched_assert.assertSwitch(
                "cluster",
                switch_from,
                switch_to,
                window=switch_window,
                rank=len(experiment.wload.tasks)),
            msg=msg)

    @experiment_test
    def test_little_big_switch1(self, experiment, tasks):
        """Wake Migration: LITTLE -> BIG: 1"""
        self._assert_switch(experiment, "big", 1)

    @experiment_test
    def test_little_big_switch2(self, experiment, tasks):
        """Wake Migration: LITTLE -> BIG: 2"""

        # little - big - little - big
        #                       ^
        # We want to test that this little to big migration happens.  So we skip
        # the first three phases.
        self._assert_switch(experiment, "big", 3)

    @experiment_test
    def test_big_little_switch1(self, experiment, tasks):
        """Wake Migration: BIG -> LITLLE: 1"""
        self._assert_switch(experiment, "little", 0)

    @experiment_test
    def test_big_little_switch2(self, experiment, tasks):
        """Wake Migration: BIG -> LITLLE: 2"""

        # little - big - little - big
        #              ^
        # We want to test that this big to little migration happens.  So we skip
        # the first two phases.
        self._assert_switch(experiment, "little", 2)
