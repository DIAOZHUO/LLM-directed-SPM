import traceback


def get_exception_message(error: BaseException, show_trace=True):
    classname = error.__class__.__name__
    message = str(error)
    if show_trace:
        trace = "trace:" + traceback.format_exc()
    else:
        trace = ""
    return "classname: %s, message: %s, " % (classname, message) + trace
