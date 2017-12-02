import inspect
import os
import time


def pv(name):
    """Return the command name and the result."""
    if "__" in name:
        raise ValueError("Double underscores not allowed for saftey reasons.")
    frame = inspect.currentframe().f_back
    val = eval(name, frame.f_globals, frame.f_locals)
    return '{0}: {1}'.format(name, val)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{0} ({1}, {2}) {3:2.2f} sec'.format(method.__name__, args, kw, te - ts))
        return result

    return timed


def timeit2(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('{0} (..., {1}) took {2:2.2f} seconds sec'.format(method.__name__, kw, te - ts))
        return result

    return timed


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))