import csv
import glob
import itertools
import os
import random
import string
import time
from string import punctuation

import fitz
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.porter import PorterStemmer
from transformers import \
    DistilBertTokenizer  # required for this function only "custom_padding()"

nltk.download("stopwords")
nltk.download("punkt")

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


nlp = spacy.load("en_core_web_sm")


def stemming(your_string: str) -> list:
    """
    Output a list of stemmed stings. Algorithm used PotterStemmer

    Example usage:
    your_string = "The best definition of man is: a being that walks on two legs and is ungrateful"

    stemming(your_string)

    output: ['the', 'best', 'definit', 'of', 'man', 'is', ':', 'a', 'be', 'that', 'walk', 'on', 'two', 'leg', 'and', 'is', 'ungrat']
    """
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

    Source: https://stackoverflow.com/a/22842533/12946268
    """

    def random_gen(lower, higher):
        while True:
            yield random.randrange(lower, higher)

    gen = random_gen(low, high)

    items = set()

    for x in itertools.takewhile(lambda x: len(items) < total_number_you_want, gen):
        items.add(x)

    return list(items)


deutsche_pipeline = spacy.load("de_core_news_sm")


def deutsche_remove_stop_words_and_punc(text: str, print_text=False) -> str:
    """
    Returns a string with stop words and punctuation removed for Deutsche text

    text: input Deutsche text

    Example usage:

    >> remove_stop_words_and_punct_de("Die etymologischen Vorformen von deutsch bedeuteten ursprünglich „zum Volk gehörig“")
    >> print(remove_stop_words_and_punct_de)
    >> etymologischen Vorformen deutsch bedeuteten ursprünglich Volk gehörig

    Source: https://github.com/cj2001/nodes2021_kg_workshop/blob/main/notebooks/00-populate_basic_graph.ipynb
    """
    result = list()
    raw_doc = deutsche_pipeline(text)
    for token in raw_doc:
        if print_text:
            print(token, token.is_stop)
            print("|-----------------|")
        if not token.is_stop and not token.is_punct:
            result.append(str(token))
    result = " ".join(result)

    return result


model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


def custom_padding(sentence_1: str, sentence_2: str):
    """
    For this required packages is "transformer"

    Compare two sentence and then pad (post) the one that has less tokens with the
    pad_token_id (defined by the model)

    model used for this example: "distilbert-base-uncased"

    Example usage:
    >>>sentence_1  = "I’ve been waiting for uploading this on github for a long time"
    >>>sentence_2  = "This is not that bad!"
    >>>print(custom_padding(sentence_1, sentence_2))
    >>> Sentence 1: [1045, 1521, 2310, 2042, 3403, 2005, 2039, 18570, 2023,
    2006, 21025, 2705, 12083, 2005, 1037, 2146, 2051], Sentence 2: [2023, 2003,
    2025, 2008, 2919, 999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    """

    s1_token = tokenizer.tokenize(sentence_1)
    s2_token = tokenizer.tokenize(sentence_2)

    s1_id = tokenizer.convert_tokens_to_ids(s1_token)
    s2_id = tokenizer.convert_tokens_to_ids(s2_token)

    if len(s2_id) < len(s1_id):
        for _ in range(len(s1_id) - len(s2_id)):
            s2_id.append(tokenizer.pad_token_id)
    else:
        for _ in range(len(s2_id) - len(s1_id)):
            s1_id.append(tokenizer.pad_token_id)

    return f"Sentence 1: {s1_id}, Sentence 2: {s2_id}"


def remove_stopwords_german(text):
    '''
    Remove stopwards from german text.

    This can also be used to clean texts in csv format

    Examople usage for a single text:
    >>>remove_stopwords_german("Jeder Satz wurde per Crowdsourcing entweder als unterstützendes Argument, als angreifendes Argument oder als kein Argument in Bezug auf das Thema kommentiert")
    >>>Satz wurde per Crowdsourcing entweder unterstützendes Argument angreifendes Argument Argument Bezug Thema kommentiert

    Example usage of text cleaning in csv format. It will remove the stopwards for all the item in the column "sentence"
    >>> train_raw['data'] = train_raw['sentence'].apply(remove_stopwords_german)


    Source: https://towardsdatascience.com/cross-topic-argument-mining-learning-how-to-classify-texts-1d9e5c00c4cc
    '''
    stpword = stopwords.words("german")
    no_punctuation = [char for char in text if char not in string.punctuation]
    no_punctuation = "".join(no_punctuation)
    return " ".join(
        [word for word in no_punctuation.split() if word.lower() not in stpword]
    )


def combining_csv(current_dir):
    '''
    Combined several csv file to one csv (must same header).

    Please change the delimeter and other arguments according to need.

    Example Usage:

    >>>combine_csv("/content/drive/MyDrive/MyDatasetCollection/argument_mining/train")
    This will create a combined csv file with name "combined_csv.tsv" on the provided location

    Reference: https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
    '''
    os.chdir(current_dir)
    extension = 'tsv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    combined_csv = pd.concat([pd.read_csv(f, header=0, delimiter="\t",
                             quoting=csv.QUOTE_NONE, encoding='utf-8') for f in all_filenames])
    combined_csv.to_csv(current_dir + "/combined_csv.tsv", index=False, encoding='utf-8-sig')
