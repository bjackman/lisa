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

from bart.common.Analyzer import Analyzer
import collections
import datetime
import gzip
import json
import os
import re
import time
import trappy

# Configure logging
import logging
reload(logging)
logging.basicConfig(
    format='%(asctime)-9s %(levelname)-8s: %(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S')

# Add support for Test Environment configuration
from env import TestEnv

# Add JSON parsing support
from conf import JsonConf

import wlgen

class Executor():

    def __init__(self, target_conf=None, tests_conf=None):
        """
        Tests Executor

        A tests executor is a module which support the execution of a
        configured set of experiments. Each experiment is composed by:
        - a target configuration
        - a worload to execute

        The executor module can be configured to run a set of workloads
        (wloads) in each different target configuration of a specified set
        (confs). These wloads and confs can be specified by the "tests_config"
        input dictionary.

        All the results generated by each experiment will be collected a result
        folder which is named according to this template:
            results/<test_id>/<wltype>:<conf>:<wload>/<run_id>
        where:
        - <test_id> : the "tid" defined by the tests_config, or a timestamp
                      based folder in case "tid" is not specified
        - <wltype>  : the class of workload executed, e.g. rtapp or sched_perf
        - <conf>    : the identifier of one of the specified configurations
        - <wload>   : the identified of one of the specified workload
        - <run_id>  : the progressive execution number from 1 up to the
                      specified iterations
        """

        # Initialize globals
        self._default_cgroup = None
        self._cgroup = None

        # Setup test configuration
        if isinstance(tests_conf, dict):
            logging.info('%14s - Loading custom (inline) test configuration',
                    'Target')
            self._tests_conf = tests_conf
        elif isinstance(tests_conf, str):
            logging.info('%14s - Loading custom (file) test configuration',
                    'Target')
            json_conf = JsonConf(tests_conf)
            self._tests_conf = json_conf.load()
        else:
            raise ValueError('test_conf must be either a dictionary or a filepath')

        # Check for mandatory configurations
        if 'confs' not in self._tests_conf or not self._tests_conf['confs']:
            raise ValueError(
                    'Configuration error: missing \'conf\' definitions')
        if 'wloads' not in self._tests_conf or not self._tests_conf['wloads']:
            raise ValueError(
                    'Configuration error: missing \'wloads\' definitions')

        # Setup devlib to access the configured target
        self.te = TestEnv(target_conf, tests_conf)
        self.target = self.te.target

        # Compute total number of experiments
        self._exp_count = self._tests_conf['iterations'] \
                * len(self._tests_conf['wloads']) \
                * len(self._tests_conf['confs'])

        self._print_section('Executor', 'Experiments configuration')

        logging.info('%14s - Configured to run:', 'Executor')

        logging.info('%14s -   %3d targt configurations:',
                     'Executor', len(self._tests_conf['confs']))
        target_confs = [conf['tag'] for conf in self._tests_conf['confs']]
        target_confs = ', '.join(target_confs)
        logging.info('%14s -       %s', 'Executor', target_confs)

        logging.info('%14s -   %3d workloads (%d iterations each)',
                     'Executor', len(self._tests_conf['wloads']),
                     self._tests_conf['iterations'])
        wload_confs = ', '.join(self._tests_conf['wloads'])
        logging.info('%14s -       %s', 'Executor', wload_confs)

        logging.info('%14s - Total: %d experiments',
                     'Executor', self._exp_count)

        logging.info('%14s - Results will be collected under:', 'Executor')
        logging.info('%14s -       %s', 'Executor', self.te.res_dir)

    def run(self):
        self._print_section('Executor', 'Experiments execution')

        # Run all the configured experiments
        exp_idx = 1
        for tc in self._tests_conf['confs']:
            # TARGET: configuration
            if not self._target_configure(tc):
                continue
            for wl_idx in self._tests_conf['wloads']:
                # TEST: configuration
                wload = self._wload_init(tc, wl_idx)
                for itr_idx in range(1, self._tests_conf['iterations']+1):
                    # WORKLOAD: execution
                    self._wload_run(exp_idx, tc, wl_idx, wload, itr_idx)
                    exp_idx += 1

        self._print_section('Executor', 'Experiments execution completed')
        logging.info('%14s - Results available in:', 'Executor')
        logging.info('%14s -       %s', 'Executor', self.te.res_dir)


################################################################################
# Target Configuration
################################################################################

    def _cgroups_init(self, tc):
        self._default_cgroup = None
        if 'cgroups' not in tc:
            return True
        if 'cgroups' not in self.target.modules:
            raise RuntimeError('CGroups module not available. Please ensure '
                               '"cgroups" is listed in your target/test modules')
        logging.info(r'%14s - Initialize CGroups support...', 'CGroups')
        errors = False
        for kind in tc['cgroups']['conf']:
            logging.info(r'%14s - Setup [%s] controller...',
                    'CGroups', kind)
            controller = self.target.cgroups.controller(kind)
            if not controller:
                logging.warning(r'%14s - CGroups controller [%s] NOT available',
                        'CGroups', kind)
                errors = True
        return not errors

    def _setup_kernel(self, tc):
        # Deploy kernel on the device
        self.te.install_kernel(tc, reboot=True)
        # Setup the rootfs for the experiments
        self._setup_rootfs(tc)

    def _setup_sched_features(self, tc):
        if 'sched_features' not in tc:
            logging.debug('%14s - Configuration not provided', 'SchedFeatures')
            return
        feats = tc['sched_features'].split(",")
        for feat in feats:
            logging.info('%14s - Set scheduler feature: %s',
                         'SchedFeatures', feat)
            self.target.execute('echo {} > /sys/kernel/debug/sched_features'.format(feat))

    def _setup_rootfs(self, tc):
        # Initialize CGroups if required
        self._cgroups_init(tc)
        # Setup target folder for experiments execution
        self.te.run_dir = os.path.join(
                self.target.working_directory, TGT_RUN_DIR)
        # Create run folder as tmpfs
        logging.debug('%14s - Setup RT-App run folder [%s]...',
                'TargetSetup', self.te.run_dir)
        self.target.execute('[ -d {0} ] || mkdir {0}'\
                .format(self.te.run_dir), as_root=True)
        self.target.execute(
                'grep schedtest /proc/mounts || '\
                '  mount -t tmpfs -o size=1024m {} {}'\
                .format('schedtest', self.te.run_dir),
                as_root=True)

    def _setup_cpufreq(self, tc):
        if 'cpufreq' not in tc:
            logging.warning(r'%14s - governor not specified, '\
                    'using currently configured governor',
                    'CPUFreq')
            return

        cpufreq = tc['cpufreq']
        logging.info(r'%14s - Configuring all CPUs to use [%s] governor',
                'CPUFreq', cpufreq['governor'])

        self.target.cpufreq.set_all_governors(cpufreq['governor'])

        if 'params' in cpufreq:
            logging.info(r'%14s - governor params: %s',
                    'CPUFreq', str(cpufreq['params']))
            for cpu in self.target.list_online_cpus():
                self.target.cpufreq.set_governor_tunables(
                        cpu,
                        cpufreq['governor'],
                        **cpufreq['params'])

    def _setup_cgroups(self, tc):
        if 'cgroups' not in tc:
            return True
        # Setup default CGroup to run tasks into
        if 'default' in tc['cgroups']:
            self._default_cgroup = tc['cgroups']['default']
        # Configure each required controller
        if 'conf' not in tc['cgroups']:
            return True
        errors = False
        for kind in tc['cgroups']['conf']:
            controller = self.target.cgroups.controller(kind)
            if not controller:
                logging.warning(r'%14s - Configuration error: '\
                        '[%s] contoller NOT supported',
                        'CGroups', kind)
                errors = True
                continue
            self._setup_controller(tc, controller)
        return not errors

    def _setup_controller(self, tc, controller):
        kind = controller.kind
        # Configure each required groups for that controller
        errors = False
        for name in tc['cgroups']['conf'][controller.kind]:
            if name[0] != '/':
                raise ValueError('Wrong CGroup name [{}]. '
                                 'CGroups names must start by "/".'\
                                 .format(name))
            group = controller.cgroup(name)
            if not group:
                logging.warning(r'%14s - Configuration error: '\
                        '[%s/%s] cgroup NOT available',
                        'CGroups', kind, name)
                errors = True
                continue
            self._setup_group(tc, group)
        return not errors

    def _setup_group(self, tc, group):
        kind = group.controller.kind
        name = group.name
        # Configure each required attribute
        group.set(**tc['cgroups']['conf'][kind][name])

    def _target_configure(self, tc):
        self._print_header('TargetConfig',
                r'configuring target for [{}] experiments'\
                .format(tc['tag']))
        self._setup_kernel(tc)
        self._setup_sched_features(tc)
        self._setup_cpufreq(tc)
        return self._setup_cgroups(tc)

    def _target_conf_flag(self, tc, flag):
        if 'flags' not in tc:
            has_flag = False
        else:
            has_flag = flag in tc['flags']
        logging.debug('%14s - Check if target conf [%s] has flag [%s]: %s',
                'TargetConf', tc['tag'], flag, has_flag)
        return has_flag


################################################################################
# Workload Setup and Execution
################################################################################

    def _wload_cpus(self, wl_idx, wlspec):
        if not 'cpus' in wlspec['conf']:
            return None
        cpus = wlspec['conf']['cpus']

        if type(cpus) == list:
            return cpus
        if type(cpus) == int:
            return [cpus]

        # SMP target (or not bL module loaded)
        if not hasattr(self.target, 'bl'):
            if 'first' in cpus:
                return [ self.target.list_online_cpus()[0] ]
            if 'last' in cpus:
                return [ self.target.list_online_cpus()[-1] ]
            return self.target.list_online_cpus()

        # big.LITTLE target
        if cpus.startswith('littles'):
            if 'first' in cpus:
                return [ self.target.bl.littles_online[0] ]
            if 'last' in cpus:
                return [ self.target.bl.littles_online[-1] ]
            return self.target.bl.littles_online
        if cpus.startswith('bigs'):
            if 'first' in cpus:
                return [ self.target.bl.bigs_online[0] ]
            if 'last' in cpus:
                return [ self.target.bl.bigs_online[-1] ]
            return self.target.bl.bigs_online
        raise ValueError('Configuration error - '
                'unsupported [{}] \'cpus\' value for [{}] '\
                'workload specification'\
                .format(cpus, wl_idx))

    def _wload_task_idxs(self, wl_idx, tasks):
        if type(tasks) == int:
            return range(tasks)
        if tasks == 'cpus':
            return range(len(self.target.core_names))
        if tasks == 'little':
            return range(len([t
                for t in self.target.core_names
                if t == self.target.little_core]))
        if tasks == 'big':
            return range(len([t
                for t in self.target.core_names
                if t == self.target.big_core]))
        raise ValueError('Configuration error - '
                'unsupported \'tasks\' value for [{}] '\
                'RT-App workload specification'\
                .format(wl_idx))

    def _wload_rtapp(self, wl_idx, wlspec, cpus):
        conf = wlspec['conf']
        logging.debug(r'%14s - Configuring [%s] rt-app...',
                'RTApp', conf['class'])

        # Setup a default "empty" task name prefix
        if 'prefix' not in conf:
            conf['prefix'] = 'task_'

        # Setup a default loadref CPU
        loadref = None
        if 'loadref' in wlspec:
            loadref = wlspec['loadref']

        if conf['class'] == 'profile':
            params = {}
            # Load each task specification
            for task_name in conf['params']:
                task = conf['params'][task_name]
                task_name = conf['prefix'] + task_name
                if task['kind'] not in wlgen.__dict__:
                    logging.error(r'%14s - RTA task of kind [%s] not supported',
                            'RTApp', task['kind'])
                    raise ValueError('Configuration error - '
                        'unsupported \'kind\' value for task [{}] '\
                        'in RT-App workload specification'\
                        .format(task))
                task_ctor = getattr(wlgen, task['kind'])
                params[task_name] = task_ctor(**task['params']).get()
            rtapp = wlgen.RTA(self.target,
                        wl_idx, calibration = self.te.calibration())
            rtapp.conf(kind='profile', params=params, loadref=loadref,
                    cpus=cpus, run_dir=self.te.run_dir)
            return rtapp

        if conf['class'] == 'periodic':
            task_idxs = self._wload_task_idxs(wl_idx, conf['tasks'])
            params = {}
            for idx in task_idxs:
                task = conf['prefix'] + str(idx)
                params[task] = wlgen.Periodic(**conf['params']).get()
            rtapp = wlgen.RTA(self.target,
                        wl_idx, calibration = self.te.calibration())
            rtapp.conf(kind='profile', params=params, loadref=loadref,
                    cpus=cpus, run_dir=self.te.run_dir)
            return rtapp

        if conf['class'] == 'custom':
            rtapp = wlgen.RTA(self.target,
                        wl_idx, calibration = self.te.calib)
            rtapp.conf(kind='custom',
                    params=conf['json'],
                    duration=conf['duration'],
                    loadref=loadref,
                    cpus=cpus, run_dir=self.te.run_dir)
            return rtapp

        raise ValueError('Configuration error - '
                'unsupported \'class\' value for [{}] '\
                'RT-App workload specification'\
                .format(wl_idx))

    def _wload_perf_bench(self, wl_idx, wlspec, cpus):
        conf = wlspec['conf']
        logging.debug(r'%14s - Configuring perf_message...',
                'PerfMessage')

        if conf['class'] == 'messaging':
            perf_bench = wlgen.PerfMessaging(self.target, wl_idx)
            perf_bench.conf(**conf['params'])
            return perf_bench

        if conf['class'] == 'pipe':
            perf_bench = wlgen.PerfPipe(self.target, wl_idx)
            perf_bench.conf(**conf['params'])
            return perf_bench

        raise ValueError('Configuration error - '\
                'unsupported \'class\' value for [{}] '\
                'perf bench workload specification'\
                .format(wl_idx))

    def _wload_conf(self, wl_idx, wlspec):

        # CPUS: setup execution on CPUs if required by configuration
        cpus = self._wload_cpus(wl_idx, wlspec)

        # CGroup: setup CGroups if requried by configuration
        self._cgroup = self._default_cgroup
        if 'cgroup' in wlspec:
            if 'cgroups' not in self.target.modules:
                raise RuntimeError('Target not supporting CGroups or CGroups '
                                   'not configured for the current test configuration')
            self._cgroup = wlspec['cgroup']

        if wlspec['type'] == 'rt-app':
            return self._wload_rtapp(wl_idx, wlspec, cpus)
        if wlspec['type'] == 'perf_bench':
            return self._wload_perf_bench(wl_idx, wlspec, cpus)


        raise ValueError('Configuration error - '
                'unsupported \'type\' value for [{}] '\
                'workload specification'\
                .format(wl_idx))

    def _wload_init(self, tc, wl_idx):
        tc_idx = tc['tag']

        # Configure the test workload
        wlspec = self._tests_conf['wloads'][wl_idx]
        wload = self._wload_conf(wl_idx, wlspec)

        # Keep track of platform configuration
        self.te.test_dir = '{}/{}:{}:{}'\
            .format(self.te.res_dir, wload.wtype, tc_idx, wl_idx)
        os.system('mkdir -p ' + self.te.test_dir)
        self.te.platform_dump(self.te.test_dir)

        # Keep track of kernel configuration and version
        config = self.target.config
        with gzip.open(os.path.join(self.te.test_dir, 'kernel.config'), 'wb') as fh:
            fh.write(config.text)
        output = self.target.execute('{} uname -a'\
                .format(self.target.busybox))
        with open(os.path.join(self.te.test_dir, 'kernel.version'), 'w') as fh:
            fh.write(output)

        return wload

    def _wload_run_init(self, run_idx):
        self.te.out_dir = '{}/{}'\
                .format(self.te.test_dir, run_idx)
        logging.debug(r'%14s - out_dir [%s]', 'Executor', self.te.out_dir)
        os.system('mkdir -p ' + self.te.out_dir)

    def _wload_run(self, exp_idx, tc, wl_idx, wload, run_idx):
        tc_idx = tc['tag']

        self._print_title('Executor', 'Experiment {}/{}, [{}:{}] {}/{}'\
                .format(exp_idx, self._exp_count,
                        tc_idx, wl_idx,
                        run_idx, self._tests_conf['iterations']))

        # Setup local results folder
        self._wload_run_init(run_idx)

        # FTRACE: start (if a configuration has been provided)
        if self.te.ftrace and self._target_conf_flag(tc, 'ftrace'):
            logging.warning('%14s - FTrace events collection enabled', 'Executor')
            self.te.ftrace.start()

        # ENERGY: start sampling
        if self.te.emeter:
            self.te.emeter.reset()

        # WORKLOAD: Run the configured workload
        wload.run(out_dir=self.te.out_dir, cgroup=self._cgroup)

        # ENERGY: collect measurements
        if self.te.emeter:
            self.te.emeter.report(self.te.out_dir)

        # FTRACE: stop and collect measurements
        if self.te.ftrace and self._target_conf_flag(tc, 'ftrace'):
            self.te.ftrace.stop()

            trace_file = self.te.out_dir + '/trace.dat'
            self.te.ftrace.get_trace(trace_file)
            logging.info(r'%14s - Collected FTrace binary trace:', 'Executor')
            logging.info(r'%14s -    %s', 'Executor',
                         trace_file.replace(self.te.res_dir, '<res_dir>'))

            stats_file = self.te.out_dir + '/trace_stat.json'
            self.te.ftrace.get_stats(stats_file)
            logging.info(r'%14s - Collected FTrace function profiling:', 'Executor')
            logging.info(r'%14s -    %s', 'Executor',
                         stats_file.replace(self.te.res_dir, '<res_dir>'))

        self._print_footer('Executor')

################################################################################
# Utility Functions
################################################################################

    def _print_section(self, tag, message):
        logging.info('')
        logging.info(FMT_SECTION)
        logging.info(r'%14s - %s', tag, message)
        logging.info(FMT_SECTION)

    def _print_header(self, tag, message):
        logging.info('')
        logging.info(FMT_HEADER)
        logging.info(r'%14s - %s', tag, message)

    def _print_title(self, tag, message):
        logging.info(FMT_TITLE)
        logging.info(r'%14s - %s', tag, message)

    def _print_footer(self, tag, message=None):
        if message:
            logging.info(r'%14s - %s', tag, message)
        logging.info(FMT_FOOTER)


################################################################################
# Globals
################################################################################

# Regular expression for comments
JSON_COMMENTS_RE = re.compile(
    '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
    re.DOTALL | re.MULTILINE
)

# Target specific paths
TGT_RUN_DIR = 'run_dir'

# Logging formatters
FMT_SECTION = r'{:#<80}'.format('')
FMT_HEADER  = r'{:=<80}'.format('')
FMT_TITLE   = r'{:~<80}'.format('')
FMT_FOOTER  = r'{:-<80}'.format('')

# vim :set tabstop=4 shiftwidth=4 expandtab
