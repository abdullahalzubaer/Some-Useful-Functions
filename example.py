from useful_functions import UsefulFunctions


@UsefulFunctions.timer
def a_function(s):
    for i in range(s):
        pass


a_function(99999)
