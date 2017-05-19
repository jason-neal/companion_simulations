import inspect


def pv(name):
    """Return the command name and the result"""
    frame = inspect.currentframe().f_back
    val = eval(name, frame.f_globals, frame.f_locals)
    return '{0}: {1}'.format(name, val)
