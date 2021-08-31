import useful_functions as uf


@uf.timer
def a_function(s):
    for _ in range(s):
        pass


a_function(9109999)


# print(uf.list_to_str([1, 2, 3], [4, 5, 6], ["ABD", "ZUB"]))


a_list = ["12", "13", "3", "44", "s"]

print(uf.join_list(" ", a_list))

print(uf.str_preproces_1("He44llo ÀÈÌÒÙỲǸẀWo323rld"))
