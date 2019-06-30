import cadeFunctions


opt = (input("Chosen Function"))
option = int(opt)


def funct(x):
    if option == 1:
        print("Summation of X[i]**2\n")
    elif option == 2:
        print("Summation of (x[i] ** 2) - (10 * np.cos((2 * np.pi * x[i]) * np.pi/180)) + 10 \n")
    elif option == 3:
        func1 = cadeFunctions.function_three(x)
    elif option == 4:
        func1 = cadeFunctions.function_four(x)
    elif option == 5:
        func1 = cadeFunctions.function_five(x)
    elif option == 6:
        func1 = cadeFunctions.function_six(x)
    elif option == 7:
        func1 = cadeFunctions.function_seven(x)
    elif option == 8:
        func1 = cadeFunctions.function_eight(x)
    elif option == 9:
        func1 = cadeFunctions.function_nine(x)
    elif option == 10:
        func1 = cadeFunctions.function_ten(x)
    elif option == 11:
        func1 = cadeFunctions.function_eleven(x)
    elif option == 12:
        func1 = cadeFunctions.function_twelve(x)
    elif option == 13:
        func1 = cadeFunctions.function_thirteen(x)
    elif option == 14:
        func1 = cadeFunctions.function_fourteen(x)
    else:
        func1 = cadeFunctions.function_fifteen(x)
    return func1
