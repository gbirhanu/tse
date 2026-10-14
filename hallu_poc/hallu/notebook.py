"""Show subprocess output through notebook Python, including Colab progress."""
import codecs
import queue
import subprocess
import threading
import time


class _ProgressIndicator:
    def __init__(self):
        self.handle = None
        self.last_update = 0.0

    def update(self, message):
        # A single display avoids one notebook output row per tqdm refresh.
        if time.monotonic() - self.last_update < 0.2:
            return
        self.last_update = time.monotonic()
        try:
            from IPython import get_ipython
            from IPython.display import display
            if get_ipython() is not None:
                data = {'text/plain': message}
                if self.handle is None:
                    self.handle = display(data, raw=True, display_id=True)
                else:
                    self.handle.update(data, raw=True)
                return
        except ImportError:
            pass
        print('\r' + message + '\033[K', end='', flush=True)


def _read_output(pipe, events):
    """Read independently so even a silent child cannot disable the timeout."""
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    line = []
    progress = False
    pending_cr = False
    try:
        while True:
            chunk = pipe.read1(4096)
            text = decoder.decode(chunk, final=not chunk)
            for char in text:
                if pending_cr:
                    pending_cr = False
                    if char == '\n':
                        events.put(('line', ''.join(line)))
                        line = []
                        progress = False
                        continue
                    if line:
                        events.put(('progress', ''.join(line)))
                    line = []
                    progress = True
                if char == '\r':
                    pending_cr = True
                elif char == '\n':
                    events.put(('line', ''.join(line)))
                    line = []
                    progress = False
                else:
                    line.append(char)
            if not chunk:
                if line:
                    events.put(('line', ''.join(line)))
                break
            if progress and line and not pending_cr:
                events.put(('progress', ''.join(line)))
    except Exception as exc:
        events.put(('error', exc))
    finally:
        events.put(('done', None))


def run_command(command, *, cwd=None, env=None, timeout=110 * 60):
    """Run an argument list, visibly stream output, and raise on failure.

    Ordinary lines go through the parent's Python stdout with immediate flush.
    Carriage-return progress refreshes update one display. The returned
    CompletedProcess.stdout contains ordinary lines (including result tables),
    without intermediate progress refreshes. Timeout or interruption kills and
    waits for the child before returning control to the notebook.
    """
    if isinstance(command, (str, bytes)):
        raise TypeError('command must be an argument list, not a shell string')
    if timeout is not None and timeout <= 0:
        raise ValueError('timeout must be positive or None')
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    events = queue.Queue()
    reader = threading.Thread(target=_read_output, args=(process.stdout, events),
                              daemon=True)
    reader.start()
    output = []
    indicator = _ProgressIndicator()

    def remaining():
        if timeout is None:
            return None
        left = timeout - (time.monotonic() - started)
        if left <= 0:
            raise subprocess.TimeoutExpired(command, timeout, output=''.join(output))
        return left

    try:
        while True:
            left = remaining()
            try:
                kind, value = events.get(timeout=min(0.1, left) if left is not None else 0.1)
            except queue.Empty:
                continue
            if kind == 'done':
                break
            if kind == 'error':
                raise value
            if kind == 'progress':
                indicator.update(value)
            else:
                print(value, flush=True)
                output.append(value + '\n')
        returncode = process.wait(timeout=remaining())
        result = subprocess.CompletedProcess(command, returncode, ''.join(output))
        result.check_returncode()
        return result
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        reader.join(timeout=1)
        if not reader.is_alive():
            process.stdout.close()
