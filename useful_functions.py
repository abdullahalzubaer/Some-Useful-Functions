import itertools
import time


def sorting_1(*args):
    """
    You are given a string and an int.
    Your task is to print all possible size replacement combinations of the string in lexicographic sorted order.

    Example usage:

    Inpute: sorting_1("ABDULLAH",2)
    """
    albo = [
        "".join(i)
        for i in itertools.combinations_with_replacement(sorted(args[0]), int(args[1]))
    ]
    print(albo)
    for i in albo:
        print(i)
        # return i


def list_to_str(*args: list) -> str:
    """
    Convert list to strings with cartesian product

    Example usage:

    Input: list_to_str([1, 2, 3], [4, 5, 6], ["ABD", "ZUB"])

    """
    return " ".join(str(e) for e in (list(itertools.product(*args, repeat=1))))


def timer(function):
    """
    A simple timer decorator to wrap around any function to get the execution time.

    Example usage:

    @UsefulFunctions.timer
    def a_function(s):
        for i in range(s):
            pass

    a_function(99999)
    """

    def wrapper(*args, **kwargs):
        before = time.time()
        function_call = function(*args, **kwargs)
        after = time.time()
        fname = function.__name__
        print(f"{fname} took {after-before} for execution")
        # return function_call

    return wrapper
