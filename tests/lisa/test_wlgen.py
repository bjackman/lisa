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

"""Base classes and utilities for self-testing LISA's wlgen packages"""

from contextlib import contextmanager, nested
import logging
import os
import shutil
from unittest import TestCase

from wlgen import Workload

from devlib import LocalLinuxTarget, Platform

dummy_calibration = {}

class TestTarget(LocalLinuxTarget):
    """
    Devlib target for self-testing LISA

    Uses LocalLinuxTarget configured to disallow using root.
    Adds facility to record the commands that were executed for asserting LISA
    behaviour.
    """
    def __init__(self, execute_callback):
        self.execute_calls = []
        self.execute_callback = execute_callback
        super(TestTarget, self).__init__(platform=Platform(),
                                         load_default_modules=False,
                                         connection_settings={'unrooted': True})

    def execute(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        self.execute_callback(*args, **kwargs)
        return super(TestTarget, self).execute(*args, **kwargs)

    @property
    def executed_commands(self):
        return [args[0] if args else kwargs['command']
                for args, kwargs in self.execute_calls]

    def clear_execute_calls(self):
        self.execute_calls = []

class WlgenSelfBase(TestCase):
    """
    Base class for wlgen self-tests

    Creates and sets up a TestTarget.

    Provides directory paths to use for output files. Deletes those paths if
    they already exist, to try and provide a clean test environment. This
    doesn't create those paths, tests should create them if necessary.
    """

    tools = []
    """Tools to install on the 'target' before each test"""

    def _execute_callback(self, cmd, *args, **kwargs):
        pass

    @property
    def target_run_dir(self):
        """Unique directory to use for creating files on the 'target'"""
        return os.path.join(self.target.working_directory,
                            'lisa_target_{}'.format(self.__class__.__name__))

    @property
    def host_out_dir(self):
        """Unique directory to use for creating files on the host"""
        return os.path.join(
            os.getenv('LISA_HOME'), 'results',
            'lisa_selftest_out_{}'.format(self.__class__.__name__))

    def setUp(self):
        self.target = TestTarget(execute_callback=self._execute_callback)
        self._log = logging.getLogger('TestWlgen')

        tools_path = os.path.join(os.getenv('LISA_HOME'),
                                  'tools', self.target.abi)
        self.target.setup([os.path.join(tools_path, tool)
                           for tool in self.tools])

        if self.target.directory_exists(self.target_run_dir):
            self.target.remove(self.target_run_dir)

        if os.path.isdir(self.host_out_dir):
            shutil.rmtree(self.host_out_dir)

        self.target.clear_execute_calls()

class DummyWorkload(Workload):
    command = 'echo DUMMY WORKLOAD'
    def conf(self, *args, **kwargs):
        self.command = DummyWorkload.command
        super(DummyWorkload, self).conf(*args, **kwargs)

class DummyContextManager(object):
    def __init__(self, identifier=None, broken_enter=False, broken_exit=False):
        self._log = logging.getLogger('TestExecutor')
        self.broken_enter = broken_enter
        self.broken_exit = broken_exit
        self.identifier = identifier
        self.state = 'UNUSED'

    def __enter__(self):
        self._log.info('entering context [{}]'.format(self.identifier))
        if self.broken_enter:
            raise BrokenManagerException(
                'INJECTED FAILURE IN __enter__ [{}]'.format(self.identifier))
        self.state = 'ENTERED'

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.info('exiting context [{}]'.format(self.identifier))
        if self.broken_exit:
            raise BrokenManagerException(
                'INJECTED FAILURE IN __exit__ [{}]'.format(self.identifier))
        self.state = 'EXITED'

class BrokenManagerException(Exception):
    pass

class TestBrokenContextExit(WlgenSelfBase):
    """
    Test that even if the inner context breaks the outer context exits

    E.g. if we want to freeze_userspace and collect ftrace, we should still
    unfreeze userspace even if we got an error when stopping ftrace collection.
    """
    def _execute_callback(self, command, *args, **kwargs):
        if command == DummyWorkload.command:
            self.assertEqual(self.outer.state, 'ENTERED')
            self.assertEqual(self.inner.state, 'ENTERED')

    def setUp(self):
        super(TestBrokenContextExit, self).setUp()
        self.outer = DummyContextManager('outer')
        self.inner = DummyContextManager('inner', broken_exit=True)
        self.ctx = nested(self.outer, self.inner)

    def runTest(self):
        wl = DummyWorkload(self.target, 'dummy_wload')
        wl.conf(kind='profile', params={}, duration=1)

        with self.assertRaises(BrokenManagerException):
            wl.run(context=self.ctx)

        self.assertEqual(self.outer.state, 'EXITED')
        self.assertEqual(self.inner.state, 'ENTERED')

class BrokenWorkloadException(Exception):
    pass

class TestBrokenWorkload(WlgenSelfBase):
    """
    Test that even if the workload fails, we still exit contexts

    E.g. if we freeze userspace, then get an error when running the workload, we
    should still unfreeze userspace.
    """
    def _execute_callback(self, command, *args, **kwargs):
        if command == DummyWorkload.command:
            self.assertEqual(self.ctx.state, 'ENTERED')
            self._log.info('Injecting workload failure')
            raise BrokenWorkloadException('INJECTED WORKLOAD FAILURE')

    def setUp(self):
        super(TestBrokenWorkload, self).setUp()
        self.ctx = DummyContextManager()

    def runTest(self):
        wl = DummyWorkload(self.target, 'dummy_wload')
        wl.conf(kind='profile', params={}, duration=1)

        with self.assertRaises(BrokenWorkloadException):
            wl.run(context=self.ctx)

        self.assertEqual(self.ctx.state, 'EXITED')
