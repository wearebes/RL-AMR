"""Independent, persistent FEM environments with explicit process ownership."""

import faulthandler
import multiprocessing as mp
import os
import traceback


def _worker(connection, source_type):
    faulthandler.enable()
    try:
        source = source_type()
        connection.send((True, source.settings))
        while True:
            command, argument = connection.recv()
            if command == "close":
                break
            result = getattr(source, command)(argument)
            connection.send((True, result))
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class FEMWorkers:
    """Send reset/step to each worker before receiving results in slot order.

    Only NumPy observations, scalars and settings cross the pipes. MFEM objects
    remain in the process that created them. num_workers=0 is a serial reference.
    """

    def __init__(self, source_type, num_workers):
        self.connections, self.processes = [], []
        self.source = source_type() if num_workers == 0 else None
        if self.source is not None:
            self.settings = self.source.settings
            return
        context = mp.get_context("spawn")
        names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in names}
        try:
            # Child imports see one CPU thread; the parent's torch setting stays unchanged.
            os.environ.update({name: "1" for name in names})
            for slot in range(num_workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, source_type),
                                          name=f"fem-{slot}")
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
            settings = [self._receive(slot) for slot in range(num_workers)]
            self.settings = settings[0]
        except BaseException:
            self.close()
            raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _receive(self, slot):
        connection, process = self.connections[slot], self.processes[slot]
        while not connection.poll(0.2):
            if not process.is_alive():
                raise RuntimeError(f"FEM worker {slot} exited ({process.exitcode})")
        try:
            success, result = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"FEM worker {slot} disconnected") from error
        if not success:
            raise RuntimeError(f"FEM worker {slot}:\n{result}")
        return result

    def _run(self, command, arguments):
        if self.source is not None:
            return {slot: getattr(self.source, command)(value)
                    for slot, value in arguments.items()}
        for slot, value in arguments.items():
            self.connections[slot].send((command, value))
        return {slot: self._receive(slot) for slot in arguments}

    def reset(self, seeds):
        return self._run("reset", dict(enumerate(seeds)))

    def step(self, actions):
        return self._run("step", actions)

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
