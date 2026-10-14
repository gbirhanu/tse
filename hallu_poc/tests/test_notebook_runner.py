"""Notebook output must be visible before completion and stop at its budget."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hallu import notebook


class NotebookRunnerTests(unittest.TestCase):
    def command(self, code):
        return [sys.executable, '-u', '-c', code]

    def test_output_streams_through_parent_before_child_exits(self):
        # The child cannot exit until the parent's stdout receives its line.
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'seen'

            class ParentOutput(io.StringIO):
                def write(self, value):
                    if 'ready' in value:
                        marker.touch()
                    return super().write(value)

            capture = ParentOutput()
            code = ("import pathlib,time; print('ready', flush=True); "
                    f"marker=pathlib.Path({str(marker)!r}); "
                    "deadline=time.monotonic()+4\n"
                    "while not marker.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
                    "assert marker.exists(), 'parent buffered output until process exit'\n"
                    "print('finished')")
            with contextlib.redirect_stdout(capture):
                result = notebook.run_command(self.command(code), timeout=8)
            self.assertEqual(result.stdout, 'ready\nfinished\n')
            self.assertEqual(capture.getvalue(), result.stdout)

    def test_merges_stderr_and_reports_failure(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture), self.assertRaises(subprocess.CalledProcessError) as caught:
            notebook.run_command(self.command("import sys; print('problem',file=sys.stderr); sys.exit(7)"))
        self.assertEqual(caught.exception.returncode, 7)
        self.assertIn('problem', capture.getvalue())
        self.assertIn('problem', caught.exception.output)

    def test_carriage_return_progress_does_not_accumulate_output_rows(self):
        capture = io.StringIO()
        code = "import sys; sys.stdout.write('start\\n' + ''.join('\\rprogress '+str(i) for i in range(100)) + '\\n| F1 |\\n| 60.2 |\\n')"
        with contextlib.redirect_stdout(capture), patch.object(notebook, '_ProgressIndicator') as indicator:
            result = notebook.run_command(self.command(code))
        self.assertIn('start\n', result.stdout)
        self.assertIn('| 60.2 |', capture.getvalue())
        self.assertEqual(capture.getvalue().count('progress'), 1)
        self.assertGreater(indicator.return_value.update.call_count, 0)

    def test_silent_child_is_killed_and_waited_on_timeout(self):
        processes = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        started = time.monotonic()
        with patch.object(notebook.subprocess, 'Popen', side_effect=launch):
            with self.assertRaises(subprocess.TimeoutExpired):
                notebook.run_command(self.command('import time; time.sleep(30)'), timeout=0.3)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(processes[0].returncode)

    def test_keyboard_interrupt_kills_and_waits_for_child(self):
        processes = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(notebook.subprocess, 'Popen', side_effect=launch), \
                patch.object(notebook._ProgressIndicator, 'update', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                notebook.run_command(self.command("import sys,time; sys.stdout.write('\\rworking'); sys.stdout.flush(); time.sleep(30)"))
        self.assertIsNotNone(processes[0].returncode)

    def test_environment_working_directory_and_partial_last_line(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = notebook.run_command(
                self.command("import os; print(os.path.basename(os.getcwd())); print(os.environ['RUNNER_TEST'], end='')"),
                cwd=directory, env={**os.environ, 'RUNNER_TEST': 'visible'})
        self.assertEqual(result.stdout, Path(directory).name + '\nvisible\n')

    def test_rejects_shell_command_strings(self):
        with self.assertRaises(TypeError):
            notebook.run_command('python -c anything')


if __name__ == '__main__':
    unittest.main()
