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

import fileinput
import json
import os
import re
from time import sleep

import logging

class Workload(object):
    """
    Base class for workload specifications

    To use this class, you'll need to instantiate it, then call :meth:`conf` on
    the instance. The details of that method are specific to each subclass.

    :param target: Devlib target to run workload on. May be None, in which case
                   an RT-App configuration file can be generated but the
                   workload cannot be run, and calibration features will be
                   missing.
    :param name: Human-readable name for the workload.
    :param calibration: CPU calibration specification. Can be obtained from
                        :meth:`RTA.calibration`.
    """

    def __init__(self,
                 target,
                 name):

        # Target device confguration
        self.target = target

        # Specific class of this workload
        self.wtype = None

        # Name of this orkload
        self.name = name

        # The dictionary of tasks descriptors generated by this workload
        self.tasks = {}

        # The cpus on which the workload will be executed
        self.cpus = None

        # The cgroup on which the workload will be executed
        # NOTE: requires cgroups to be properly configured and associated
        #       tools deployed on the target
        self.cgroup = None

        # The command to execute a workload (defined by a derived class)
        self.command = None

        # The workload executor, to be defined by a subclass
        self.executor = None

        # Output messages generated by commands executed on the device
        self.output = {}

        # Derived clasess callback methods
        self.steps = {
            'postrun': None,
        }

        # Task specific configuration parameters
        self.duration = None
        self.run_dir = None
        self.exc_id = None

        # Setup kind specific parameters
        self.kind = None

        # Scheduler class configuration
        self.sched = None

        # Map of task/s parameters
        self.params = {}

        # Setup logging
        self._log = logging.getLogger('Workload')

        self._log.info('Setup new workload %s', self.name)

    def __callback(self, step, **kwords):
        if step not in self.steps.keys():
            raise ValueError('Callbacks for [%s] step not supported', step)
        if self.steps[step] is None:
            return
        self._log.debug('Callback [%s]...', step)
        self.steps[step](kwords)

    def setCallback(self, step, func):
        """
        Add a callback to be called during an execution stage.

        Intended for use by subclasses. Only one callback can exist for each
        stage. Available callback stages are:

          "postrun"
            Called after the workload has finished executing, unless it's being
            run in the background. Receives a ``params`` dictionary with
            ``params["destdir"]`` set to the host directory to store workload
            output in.

        :param step: Name of the step at which to call the callback.
        :param func: Callback function.
        """
        self._log.debug('Setup step [%s] callback to [%s] function',
                        step, func.__name__)
        self.steps[step] = func

    def getCpusMask(self, cpus=None):
        mask = 0x0
        for cpu in (cpus or self.target.list_online_cpus()):
            mask |= (1 << cpu)
        self._log.debug('CPUs mask for %s: 0x%X', cpus, mask)
        return mask

    def conf(self,
             kind,
             params,
             duration,
             cpus=None,
             sched={'policy': 'OTHER'},
             run_dir=None,
             exc_id=0):
        """Configure workload. See documentation for subclasses"""

        self.cpus = cpus
        self.sched = sched
        self.duration = duration
        self.run_dir = run_dir
        self.exc_id = exc_id

        # Setup kind specific parameters
        self.kind = kind

        # Map of task/s parameters
        self.params = {}

        # Initialize run folder
        if self.run_dir is None:
            self.run_dir = self.target.working_directory
        self.target.execute('mkdir -p {}'.format(self.run_dir))

        # Configure a profile workload
        if kind == 'profile':
            self._log.debug('Configuring a profile-based workload...')
            self.params['profile'] = params

        # Configure a custom workload
        elif kind == 'custom':
            self._log.debug('Configuring custom workload...')
            self.params['custom'] = params

        else:
            self._log.error('%s is not a supported RTApp workload kind', kind)
            raise ValueError('RTApp workload kind not supported')

    def run(self,
            ftrace=None,
            cgroup=None,
            cpus=None,
            background=False,
            out_dir='./',
            as_root=True,
            start_pause_s=None,
            end_pause_s=None):
        """
        This method starts the execution of the workload. If the user provides
        an ftrace object, the method will also collect a trace.

        :param ftrace: FTrace object to collect a trace. If using
            :class:`TestEnv`, you can use :attr:`TestEnv.ftrace` for this.
        :type ftrace: :mod:`devlib.trace.FTraceCollector`

        :param cgroup: specifies the cgroup name in which the workload has to
                       run
        :type cgroup: str

        :param cpus: the CPUs on which to run the workload.

                     .. note:: if specified it overrides the CPUs specified at
                               configuration time
        :type cpus: list(int)

        :param background: run the workload in background. In this case the
                           method will not return a result. When used with
                           ftrace it is up to the caller to stop trace
                           collection
        :type background: bool

        :param out_dir: output directory where to store the collected trace or
                        other workload report (if any)
        :type out_dir: str

        :param as_root: run the workload as root on the target
        :type as_root: bool

        :param start_pause_s: time to wait before executing the workload in
                              seconds. If ftrace is provided, trace collection
                              is started before waiting.
        :type start_pause_s: float

        :param end_pause_s: time to wait after executing the workload in
                            seconds. If ftrace is provided, trace collection is
                            stopped after this wait time.
        :type end_pause_s: float
        """

        self.cgroup = cgroup

        # Compose the actual execution command starting from the base command
        # defined by the base class
        _command = self.command

        if not _command:
            self._log.error('Error: empty executor command')

        # Prepend eventually required taskset command
        if cpus or self.cpus:
            cpus_mask = self.getCpusMask(cpus if cpus else self.cpus)
            taskset_cmd = '{}/taskset 0x{:X}'\
                    .format(self.target.executables_directory,
                            cpus_mask)
            _command = '{} {}'\
                    .format(taskset_cmd, _command)

        if self.cgroup:
            if hasattr(self.target, 'cgroups'):
                _command = self.target.cgroups.run_into_cmd(self.cgroup,
                                                            _command)
            else:
                raise ValueError('To run workload in a cgroup, add "cgroups" '
                                 'devlib module to target/test configuration')

        # Start FTrace (if required)
        if ftrace:
            ftrace.start()

        # Wait `start_pause` seconds before running the workload
        if start_pause_s:
            self._log.info('Waiting %f seconds before starting workload execution',
                           start_pause_s)
            sleep(start_pause_s)

        # Start task in background if required
        if background:
            self._log.debug('WlGen [background]: %s', _command)
            self.target.background(_command, as_root=as_root)
            self.output['executor'] = ''

        # Start task in foreground
        else:
            self._log.info('Workload execution START:')
            self._log.info('   %s', _command)
            # Run command and wait for it to complete
            results = self.target.execute(_command, as_root=as_root)
            self.output['executor'] = results

        # Wait `end_pause` seconds before stopping ftrace
        if end_pause_s:
            self._log.info('Waiting %f seconds before stopping trace collection',
                           end_pause_s)
            sleep(end_pause_s)

        # Stop FTrace (if required)
        ftrace_dat = None
        if ftrace:
            ftrace.stop()
            ftrace_dat = out_dir + '/' + self.test_label + '.dat'
            dirname = os.path.dirname(ftrace_dat)
            if not os.path.exists(dirname):
                self._log.debug('Create ftrace results folder [%s]',
                                dirname)
                os.makedirs(dirname)
            self._log.info('Pulling trace file into [%s]...', ftrace_dat)
            ftrace.get_trace(ftrace_dat)

        if not background:
            self.__callback('postrun', destdir=out_dir)
            self._log.debug('Workload execution COMPLETED')

        return ftrace_dat

    def getOutput(self, step='executor'):
        return self.output[step]

    def listAll(self, kill=False):
        # Show all the instances for the current executor
        tasks = self.target.run('ps | grep {0:s}'.format(self.executor))
        for task in tasks:
            task = task.split()
            self._log.info('%5s: %s (%s)', task[1], task[8], task[0])
            if kill:
                self.target.run('kill -9 {0:s}'.format(task[1]))

    def killAll(self):
        if self.executor is None:
            return
        self._log.info('Killing all [%s] instances:', self.executor)
        self.listAll(True)
