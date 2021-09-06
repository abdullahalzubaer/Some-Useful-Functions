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


corpus = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis ultricies condimentum dui eget viverra. Phasellus ut pharetra justo. Curabitur interdum venenatis nisi vel aliquam. Donec auctor, tortor ac tristique convallis, felis turpis varius nisl, ac elementum neque leo ac quam. Proin in mattis risus. In eu dolor nulla. Praesent eleifend laoreet interdum. Nam pellentesque tellus ut est hendrerit, eget ornare ipsum interdum. Donec imperdiet, nisl sed bibendum vulputate, nulla ante lobortis sapien, eu gravida neque libero vitae massa."""

print(uf.str_count("your corpus here", "here"))

string_list = ["ABC", "DEF", "GEMM", "ababv", "asdwd", "ABscw"]

print(uf.str_remove_lowercase(string_list))

print(uf.str_remove_uppercase(string_list))
