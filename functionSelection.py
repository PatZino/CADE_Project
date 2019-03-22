import cadeFunctions


print("Functions")
option = input("Select a function by entering a number: ")


def funct(x):
    if option == 1:
        func1 = cadeFunctions.function_one(x)
    elif option == 2:
        func1 = cadeFunctions.function_two(x)
    elif option == 3:
        func1 = cadeFunctions.function_three(x)
    elif option == 4:
        func1 = cadeFunctions.function_four(x)
    elif option == 2:
        func1 = cadeFunctions.function_five(x)
    else:
        func1 = cadeFunctions.function_three(x)
    return func1
