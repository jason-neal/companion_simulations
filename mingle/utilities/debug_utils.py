import inspect
import os
import time

from typing import Callable


def pv(name: str) -> str:
    """Return the command name and the result."""
    if "__" in name:
        raise ValueError("Double underscores not allowed for saftey reasons.")
    frame = inspect.currentframe().f_back
    val = eval(name, frame.f_globals, frame.f_locals)
    return '{0}: {1}'.format(name, val)


def timeit(method: Callable) -> Callable:
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{0} ({1}, {2}) {3:2.2f} sec'.format(method.__name__, args, kw, te - ts))
        return result

    return timed


def timeit2(method: Callable) -> Callable:
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{0} (..., {1}) took {2:2.2f} seconds'.format(method.__name__, kw, te - ts))
        return result

    return timed


def list_files(startpath: str) -> None:
    for root, _, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{0}{1}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{0}{1}'.format(subindent, f))
