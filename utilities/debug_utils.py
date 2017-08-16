import inspect
import time


def pv(name):
    """Return the command name and the result"""
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

        print('{0} ({1}, {2}) {3:2.2f} sec'.format(method.__name__, args, kw, te-ts))
        return result

    return timed
