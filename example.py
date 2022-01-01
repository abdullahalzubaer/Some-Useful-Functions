import pandas as pd
import seaborn as sns

import useful_functions as uf


@uf.timer
def a_function(s):
    for _ in range(s):
        pass


a_function(9109999)


# print(uf.list_to_str([1, 2, 3], [4, 5, 6], ["ABD", "ZUB"]))


a_list = ["12", "13", "3", "44", "s"]

# print(uf.join_list(" ", a_list))

# print(uf.str_preproces_1("He44llo ÀÈÌÒÙỲǸẀWo323rld"))


corpus = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis ultricies condimentum dui eget viverra. Phasellus ut pharetra justo. Curabitur interdum venenatis nisi vel aliquam. Donec auctor, tortor ac tristique convallis, felis turpis varius nisl, ac elementum neque leo ac quam. Proin in mattis risus. In eu dolor nulla. Praesent eleifend laoreet interdum. Nam pellentesque tellus ut est hendrerit, eget ornare ipsum interdum. Donec imperdiet, nisl sed bibendum vulputate, nulla ante lobortis sapien, eu gravida neque libero vitae massa."""

# print(uf.str_count("your corpus here", "here"))

string_list = ["ABC", "DEF", "GEMM", "ababv", "asdwd", "ABscw"]

# print(uf.str_remove_lowercase(string_list))

# print(uf.str_remove_uppercase(string_list))


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# print(uf.standard_deviation(numbers))


# to_str = uf.pdf_to_str("rand.pdf")

# print(to_str)


# file_location = "FILE LOCATION"
#
# df_original = pd.read_csv(file_location)
# df_class_label = df_original[["class_label"]]
#
# uf.class_distribution("class_label", df_class_label)

# print(
#     uf.stemming(
#         "The best definition of man is: a being that walks on two legs and is ungrateful"
#     )
# )

# print(uf.random_int_generator(1, 31, 6))


print(
    uf.deutsche_remove_stop_words_and_punc(
        "Die etymologischen Vorformen von deutsch bedeuteten ursprünglich „zum Volk gehörig“"
    )
)
