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

import json
import time
import re
import pandas
import StringIO

from unittest import SkipTest

from env import TestEnv
from test import LisaTest

"""
Goal
====

Check that the configuration of a given device is suitable
for running EAS.

Detailed Description
====================

This test reads the kernel configuration and digs around in sysfs to
check the following attributes are true:
    * the minimum set of required config options are enabled
    * all CPUs have access to the 'sched' CPUFreq governor
    * energy aware scheduling is present and enabled

Expected Behaviour
==================

All required config options are set, sched governor is present.

"""

TEST_CONF = {
    'modules': ['cpufreq'],
    'results_dir': 'PreliminaryTests',
    'tools': [
        'sysbench',
    ]
}

class BasicCheckTest(LisaTest):
    @classmethod
    def setUpClass(cls):
        cls.env = TestEnv(test_conf=TEST_CONF)
        cls.target = cls.env.target

class TestSchedGovernor(BasicCheckTest):
    def test_sched_governor_available(self):
        """
        Check that the 'sched' or 'schedutil' cpufreq governor is available
        """
        fail_list = []
        for cpu in self.target.list_online_cpus():
            governors = self.target.cpufreq.list_governors(cpu)
            if 'sched' not in governors and 'schedutil' not in governors:
                fail_list.append(cpu)
        msg = 'CPUs {} do not support sched[util] cpufreq governor'.format(
            fail_list)
        self.assertTrue(len(fail_list) == 0, msg=msg)

class TestKernelConfig(BasicCheckTest):
    def test_kernel_config(self):
        """
        Check that the kernel config has the basic requirements for EAS
        """
        kernel_config = self.target.config
        if not kernel_config.text:
            raise SkipTest('Kernel config not available on target')

        # NB: We don't test for schedtune/schedutil, that's tested by
        # TestSchedGovernor.
        necessary_configs = [
            # 'CONFIG_CPU_FREQ_STAT',
            'CONFIG_CGROUPS',
            'CONFIG_SMP',
            'CONFIG_SCHED_MC',
            'CONFIG_CPU_FREQ',
            'CONFIG_CPU_IDLE',
            'CONFIG_SCHED_DEBUG',
        ]

        fail_list = [c for c in necessary_configs
                     if not kernel_config.is_enabled(c)]

        if len(fail_list):
            message = 'Missing kernel configs: ' + ', '.join(fail_list)
            self.assertTrue(len(fail_list) == 0, msg=message)

class TestWorkThroughput(BasicCheckTest):
    """
    Check that compute throughput increases with CPU frequency

    That is, check that cpufreq really works in that setting a higher
    frequency provides greater CPU performance
    """
    def _run_sysbench_work(self, cpu, duration):
        """
        Run benchmark using 1 thread on a given CPU.

        :param cpu: cpu to run the benchmark on
        :type cpu: str
        :param duration: length of time, in seconds to run the benchmark

        :returns: float - performance score
        """
        args = '--test=cpu --num-threads=1 --max-time={} run'.format(duration)

        sysbench = self.target.path.join(self.target.executables_directory,
                                         'sysbench')
        bench_out = self.target.invoke(sysbench, args=args, on_cpus=[cpu])

        match = re.search(r'(total number of events:\s*)([\d.]*)', bench_out)
        return float(match.group(2))

    def _check_work_throughput(self, cpu):
        seconds = 1.0
        margin = 0.2
        frequencies = self.target.cpufreq.list_frequencies(cpu)
        if len(frequencies) == 1:
            return True
        original_governor = self.target.cpufreq.get_governor(cpu)
        original_freq = None
        if original_governor == 'userspace':
            original_freq = self.target.cpufreq.get_frequency(cpu)
        # set userspace governor
        self.target.cpufreq.set_governor(cpu, 'userspace')
        # do each freq in turn
        result = {}
        for freq in frequencies:
            self.target.cpufreq.set_frequency(cpu, freq)
            result[freq] = self._run_sysbench_work(cpu, seconds)
        # restore governor
        self.target.cpufreq.set_governor(cpu, original_governor)
        if original_freq:
            self.target.cpufreq.set_frequency(cpu, original_freq)
        # compare work throughput
        return result[frequencies[0]] < result[frequencies[-1]]

    def test_work_throughput(self):
        failed_cpus = []
        for cpulist in self.env.topology.get_level('cpu'):
            cpu = cpulist[0]
            if not self._check_work_throughput(cpu):
                failed_cpus.append(cpu)
        msg='Work done did not scale with CPU Freq on CPUs: {}'\
            .format(failed_cpus)
        self.assertFalse(len(failed_cpus), msg=msg)

class TestEnergyModelPresent(BasicCheckTest):
    def test_energy_model_present(self):
        """Test that we can see the energy model in sysctl"""
        if not self.target.file_exists(
                '/proc/sys/kernel/sched_domain/cpu0/domain0/group0/energy/'):
            raise AssertionError(
                'No energy model visible in procfs. Possible causes: \n'
                '- Kernel built without (CONFIG_SCHED_DEBUG && CONFIG_SYSCTL)\n'
                '- No energy model in kernel')

class TestSchedutilTunables(BasicCheckTest):
    MAX_UP_RATE_LIMIT_US = 20 * 1e3
    MAX_DOWN_RATE_LIMIT_US = 20 * 1e3

    def test_rate_limit_not_too_high(self):
        """Test that the schedutil ratelimiting is not too harsh"""
        governors = self.target.cpufreq.list_governors(0)
        if 'schedutil' not in governors:
            raise SkipTest('schedutil not present on target')
        self.target.cpufreq.set_all_governors('schedutil')

        cpus = set(range(self.target.number_of_cpus))
        up_limit_fail_cpus = []
        down_limit_fail_cpus = []
        while cpus:
            cpu = iter(cpus).next()
            domain = tuple(self.target.cpufreq.get_domain_cpus(cpu))

            tunables = self.target.cpufreq.get_governor_tunables(cpu)
            if int(tunables['up_rate_limit_us']) > self.MAX_UP_RATE_LIMIT_US:
                up_limit_fail_cpus += domain
            if int(tunables['down_rate_limit_us']) > self.MAX_DOWN_RATE_LIMIT_US:
                down_limit_fail_cpus += domain

            cpus = cpus.difference(domain)

        self.assertTrue(
            up_limit_fail_cpus == [],
            'schedutil up_rate_limit_us greater than {} on CPUs {}. '
            'Responsiveness will be affected.'.format(
                self.MAX_UP_RATE_LIMIT_US, up_limit_fail_cpus))
        self.assertTrue(
            down_limit_fail_cpus == [],
            'schedutil down_rate_limit_us greater than {} on CPUs {}. '
            'Responsiveness will be affected.'.format(
                self.MAX_DOWN_RATE_LIMIT_US, down_limit_fail_cpus))

class TestSchedDomainFlags(BasicCheckTest):
    """Test requirements of sched_domain flags"""

    # See include/linux/sched.h in an EAS kernel
    SD_SHARE_CAP_STATES = 0x8000

    def setUp(self):
        if not self.target.file_exists('/proc/sys/kernel/sched_domain/'):
            raise SkipTest('sched_domain info not exposed in procfs. '
                           'Enable CONFIG_SCHED_DEBUG in target kernel')

    def iter_cpu_sd_flags(self, cpu):
        """
        Get the flags for a given CPU's sched_domains

        :param cpu: Logical CPU number whose sched_domains' flags we want
        :returns: Iterator over the flags, as an int, of each of that CPU's
                  domains, highest-level (i.e. typically "DIE") first.
        """
        base_path = '/proc/sys/kernel/sched_domain/cpu{}/'.format(cpu)
        for domain in sorted(self.target.list_directory(base_path), reverse=True):
            flags_path = self.target.path.join(base_path, domain, 'flags')
            yield self.target.read_int(flags_path)

    def test_share_cap_states(self):
        """
        Check that some domain exists with SD_SHARE_CAP_STATES set

        EAS silently does nothing if this flag is not set at any level (see
        use of sd_scs percpu variable in scheduler code).
        """
        cpu0_flags = []
        for flags in self.iter_cpu_sd_flags(0):
            if flags & self.SD_SHARE_CAP_STATES:
                return
            cpu0_flags.append(flags)
        flags_str = ', '.join([hex(f) for f in cpu0_flags])
        raise AssertionError('No sched_domain with SD_SHARE_CAP_STATES flag. '
                             'flags: {}'.format(flags_str))
