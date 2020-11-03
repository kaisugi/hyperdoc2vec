import collections
import os


def simplify_filename(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el