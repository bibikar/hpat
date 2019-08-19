# -*- coding: utf-8 -*-
import codecs
import functools
import itertools
import sys
import timeit

from numba import njit, types

import hpat

STRING_CASES = [
    'ascii',
    '1234567890',
    '¡Y tú quién te crees?',
    '🐍⚡',
    '大处着眼，小处着手。',
]

# We want to be able to print lines containing Unicode
utf8writer = codecs.getwriter('utf-8')(sys.stdout.buffer,
                                       errors='backslashreplace')
print = functools.partial(print, file=utf8writer)


def time_func(f, inner_loops=100, outer_loops=5, warmup=True,
              args=(), kwargs={}):
    '''Time execution of a function.

    The mean time taken to execute f(*args, **kwargs) over inner_loops
    iterations is measured outer_loops times.

    Parameters
    ----------
    f : callable f(*args, **kwargs)
        The function to time.
    inner_loops : int, default 100
        Number of inner loop iterations to take the mean over.
    outer_loops : int, default 5
        Number of outer loop iterations.
    warmup : bool, default True
        Whether to include an extra warm up round before timing.
    args : tuple, default ()
        Positional arguments to pass to f.
    kwargs : dict, default {}
        Keyword arguments to pass to f.

    Returns
    -------
    val
        The last value returned from f
    times : list
        The mean times
    '''

    # Warm-up iteration
    val = f(*args, **kwargs)
    times = []

    for _ in range(outer_loops):
        t0 = timeit.default_timer()
        for _ in range(inner_loops):
            val = f(*args, **kwargs)
        t1 = timeit.default_timer()

        times.append((t1 - t0) / inner_loops)

    return val, times


def benchmark(*args, **kwargs):
    def _benchmark(pyfunc):
        def __benchmark():

            # Time compilation
            cfunc, compile_times = time_func(njit, args=(pyfunc,))
            # TODO: report times
            print('compilation took %s seconds' % min(compile_times))

            for elem in itertools.product(*args, *kwargs.values()):
                # Construct arguments
                nargs = elem[:len(args)]
                nkwargs = dict(zip(kwargs.keys(), elem[len(args):]))

                # Time execution of pyfunc and cfunc
                pyval, py_times = time_func(pyfunc, args=nargs, kwargs=nkwargs)
                cval, c_times = time_func(cfunc, args=nargs, kwargs=nkwargs)

                # TODO: this is all for the assertion - better way to do this?
                nkwargs_str = tuple('%s=%r' % it for it in nkwargs.items())
                nargs_str = tuple(repr(x) for x in nargs)
                call_str = '%s(%s)' % (pyfunc.__qualname__,
                                       ', '.join(nargs_str + nkwargs_str))
                assert pyval == cval, '%s -> %s but jitted %s -> %s' % \
                                      (call_str, pyval, call_str, cval)

                # TODO: report times
                print('pyfunc %s took %s seconds' % (call_str, min(py_times)))
                print('cfunc %s took %s seconds' % (call_str, min(c_times)))
        return __benchmark
    return _benchmark


@benchmark(STRING_CASES, range(-3, 20))
def time_center(x, y):
    return x.center(y)


@benchmark(STRING_CASES, range(-3, 20), [' ', '+', 'ú', '处'])
def time_center_fillchar(x, y, fillchar):
    return x.center(y, fillchar)


@benchmark(STRING_CASES, range(-3, 20))
def time_ljust(x, y):
    return x.ljust(y)


@benchmark(STRING_CASES, range(-3, 20), [' ', '+', 'ú', '处'])
def time_ljust_fillchar(x, y, fillchar):
    return x.ljust(y, fillchar)


@benchmark(STRING_CASES, range(-3, 20))
def time_rjust(x, y):
    return x.rjust(y)


@benchmark(STRING_CASES, range(-3, 20), [' ', '+', 'ú', '处'])
def time_rjust_fillchar(x, y, fillchar):
    return x.rjust(y, fillchar)


if __name__ == "__main__":
    time_center()
    time_center_fillchar()
    time_ljust()
    time_ljust_fillchar()
    time_rjust()
    time_rjust_fillchar()
