import itertools
import random
import time

import fitz
import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
import spacy
from nltk.stem.porter import *

plt.style.use("ggplot")


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
        return function_call

    return wrapper


def join_list(deli, lst) -> str:
    """
    Join a list of string to a string using your own delimeter.

    Example usage:

    a_list = ["12", "13", "3", "44", "s"]

    print(uf.join_list(" ", a_list))

    output: "12 13 44 s"

    """

    str_returned = deli.join(lst)
    return str_returned


def str_preproces_1(input_string: str) -> str:
    """
    Return pre-processed string from input string

    Example usage:

    a_non_ascii_digit string = "He43llo ÀÈÌÒÙỲǸẀWo323rld"

    print(uf.str_preprocess(a_non_ascii_digit))

    output: "Hello World"

    """
    return "".join(c for c in input_string if c.isascii() and not c.isdigit())


def str_count(corpus: str, search_string: str) -> int:
    """
    Return frequency of a string present in the current corpus.

    Example usage:

    corpus = "your corpus here"

    count = uf.str_count(corpus, "HERE")

    output = 1

    """
    count = corpus.count(search_string)
    return count


def str_remove_lowercase(x: list) -> list:
    """
    Return a list containing string if not lower case, using filter function

    Example usage:

    string_list =["ABC", "DEF", "GEMM", "ababv", "asdwd", "ABscw"]

    print(uf.str_remove_uppercase(string_list))

    output = ['ABC', 'DEF', 'GEMM']

    Notes: Filter -> Constructs iterator from those elements of iterable for
    which function returns true
    """
    return list(filter(lambda x: x.isupper(), x))


def str_remove_uppercase(x: list) -> list:
    """
    Return a list containing string of lowercase strings using filterfalse() function

    Example usage:

    string_list =["ABC", "DEF", "GEMM", "ababv", "asdwd", "ABscw"]

    print(uf.str_remove_uppercase(string_list))

    output =['ababv', 'asdwd', 'ABscw']

    Notes: itertools.filterfalse -> Return the elements which
    returns False (opposite of filter function)

    """
    return list(itertools.filterfalse(lambda x: x.isupper(), x))


def standard_deviation(number: list) -> float:
    """
    Return the standard deviation of a list of numbers

    Example usage:

    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    print(uf.standard_deviation(numbers))
    output = 3.162
    """
    return round((np.std(number)), 3)


def pdf_to_str(document_path: str) -> str:
    """
    Convert pdf documents to strings.

    Example usage:

    to_str = uf.pdf_to_str("rand.pdf")

    output: a long string of all the pdf contents! :)


    """
    with fitz.open(document_path) as doc:
        text = str()
        for page in doc:
            text += page.getText()
    return text


def class_distribution(column_name, class_label_series):
    """
    Output a plot showing the class distribution.


    Example usage:
    file_location = "FILE_LOCATION"
    df_original = pd.read_csv(file_location)
    df_class_label = df_original[["class_label"]]

    uf.class_distribution("class_label", df_class_label)
    """
    sns.countplot(x=column_name, data=class_label_series)
    plt.show()


def stemming(your_string: str) -> list:
    """
    Output a list of stemmed stings. Algorithm used PotterStemmer

    Example usage:
    your_string = "The best definition of man is: a being that walks on two legs and is ungrateful"

    stemming(your_string)

    output: ['the', 'best', 'definit', 'of', 'man', 'is', ':', 'a', 'be', 'that', 'walk', 'on', 'two', 'leg', 'and', 'is', 'ungrat']
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(your_string)
    return [PorterStemmer().stem(token.text) for token in doc]


def random_int_generator(low: int, high: int, total_number_you_want: int) -> list:
    """
    Return a list of integer number based on uers given limit

    low: lower int, including.
    high: higher int, excluding
    total_number_you_want: Total unique random int you want

    Example usage:

    >> your_rand_int_list = random_int_generator(1, 55, 6)
    >> print(your_rand_int_list)
    >> [46, 47, 17, 51, 21, 22]
    """

    def random_gen(lower, higher):
        while True:
            yield random.randrange(lower, higher)

    gen = random_gen(low, high)

    items = set()

    for x in itertools.takewhile(lambda x: len(items) < total_number_you_want, gen):
        items.add(x)

    return list(items)
