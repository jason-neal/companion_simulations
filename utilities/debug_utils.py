import inspect


def pv(name):
    """Return the command name and the result"""
    if "__" in name:
        raise ValueError("Double underscores not allowed for saftey reasons.")
    frame = inspect.currentframe().f_back
    val = eval(name, frame.f_globals, frame.f_locals)
    return '{0}: {1}'.format(name, val)
