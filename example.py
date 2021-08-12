import useful_functions as uf


@uf.timer
def a_function(s):
    for _ in range(s):
        pass


a_function(99999)


print(uf.list_to_str([1, 2, 3], [4, 5, 6], ["ABD", "ZUB"]))
