"""Persistent FEM processes; each slot owns its environment. 0 runs serially."""

import multiprocessing as mp
import os


def _worker(connection, source_type):
    with connection:
        source = source_type()
        connection.send(source.settings)
        while True:
            command, value = connection.recv()
            connection.send(getattr(source, command)(value))


class FEMWorkers:
    def __init__(self, source_type, num_workers):
        self.connections, self.processes = [], []
        self.source = source_type() if num_workers == 0 else None
        if num_workers == 0:
            self.settings = self.source.settings
            return
        context = mp.get_context("spawn")
        names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in names}
        os.environ.update({name: "1" for name in names})
        try:
            for _ in range(num_workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, source_type))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            settings = [connection.recv() for connection in self.connections]
            self.settings = settings[0]
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _run(self, command, arguments):
        if self.source is not None:
            return {slot: getattr(self.source, command)(value)
                    for slot, value in arguments.items()}
        for slot, value in arguments.items():
            self.connections[slot].send((command, value))
        return {slot: self.connections[slot].recv() for slot in arguments}

    def reset(self, seeds):
        return self._run("reset", dict(enumerate(seeds)))

    def step(self, actions):
        return self._run("step", actions)

    def close(self):
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            process.join()
        for connection in self.connections:
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
