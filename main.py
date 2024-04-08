"""

Group: Chaya Mizrachi ID: 214102584,
        Yael Siman-Tov ID:325181295,
        Linoy Nisim Pur ID: 324029685
        Shaindel Saymon  ID: 214223299
Source Git: lihiSabag https://github.com/lihiSabag/Numerical-Analysis-2023.git
GitHub of this project: https://github.com/YaelSimanTov/Work4_B_AnalizaNomerit.git
"""

from math import e
import math
from sympy.utilities.lambdify import lambdify
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify
from sympy import lambdify, symbols, sin

import sympy as sp
from sympy.utilities.lambdify import lambdify
x = sp.symbols('x')
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[90m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
     # Background colors:
    GREYBG = '\033[100m'
    REDBG = '\033[101m'
    GREENBG = '\033[102m'
    YELLOWBG = '\033[103m'
    BLUEBG = '\033[104m'
    PINKBG = '\033[105m'
    CYANBG = '\033[106m'

def romberg_integration1(func, a, b, n, epsilon):
    """
    This function implements Romberg method by predefined level of accuracy (and not by a number of steps only)
    :param func: The function to be integrated.
    :param a: The lower limit of integration.
    :type a: float
    :param b: The upper limit of integration.
    :type b: float
    :param n:  The number of iterations (higher value leads to better accuracy).
    :type n: int
    :param epsilon: The level of accuracy of the error for which we will stop performing an integral approximation calculation
    :type epsilon: float
    Return:The approximate definite integral of the function over [a, b].
    :rtype: list of lists - 2D matrix
    """
    h = b - a
    R = []
    temp = []
    temp.append(0.5 * h * (func(a) + func(b)))  # T(1))
    R.append(temp)
    h/=2
    temp = []
    temp.append(0.5 * R[0][0] + h * func(a + h))  # T(0.5)
    temp.append(temp[0] + (1 / 3) * (temp[0] - R[0][0]))  # T(1, 0.5)
    R.append(temp)
    i = 2
    real_steps = 0
    while abs(R[i-1][-1] - R[i-2][-1]) > epsilon and i < n:
        temp = []
        h /= 2

        sum_term = 0
        for k in range(1, 2 ** i, 2):
            sum_term += func(a + k * h)

        temp.append(0.5 * R[i - 1][0] + h * sum_term)

        for j in range(1, i+1):
            temp.append(temp[j - 1] + (temp[j - 1] - R[i - 1][j - 1]) / ((4 ** j) - 1))
        R.append(temp)
        real_steps = i
        i += 1
    print("The number of steps to perform in order to approximate an integral with an error of:%f is: %d" % (epsilon, real_steps))
    return R[-1][-1]



def my_romberg_integration(func, a, b, n):
    """
    Romberg Integration

    Parameters:
    func (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of iterations (higher value leads to better accuracy).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    h = b - a
    R = np.zeros((n+1, n+1), dtype=float)

    R[0, 0] = 0.5 * h * (func(a) + func(b))
    powerOf2 = 1

    for i in range(1, n+1):
        h /= 2
        sum_term = 0
        powerOf2 = 2 * powerOf2

        for k in range(1, powerOf2, 2):
            sum_term += func(a + k * h)

        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_term
        powerOf4 = 1

        for j in range(1, i + 1):
            powerOf4 = 4 * powerOf4

            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / ( powerOf4 - 1)

        """
        n = len(R) - 1  # Assuming R is a square matrix
        for i in range(1, n + 1):
            for j in range(1, i + 1):
                print(f"R[{i}][{j}] = {R[i][j]}")
        """
    return R[n][n]





def simpsons_rule(f, a, b, n, tf):
    """
    Simpson's Rule for Numerical Integration

    Parameters:
    f (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of subintervals (must be even).
    tf (bool): Variable to decide whether to perform Error evaluation
    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    if n % 2 != 0:
        raise ValueError("Number of subintervals (n) must be even for Simpson's Rule.")

    h = (b - a) / n
    if tf:
       print(bcolors.FAIL, "Error evaluation En = ", round(SimpsonError(func(), b, a, h), 6), bcolors.ENDC)

    integral = f(a) + f(b)  # Initialize with endpoints

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0:
            integral += 2 * f(x_i)
        else:
            integral += 4 * f(x_i)

    integral *= h / 3

    return integral




def trapezoidal(a, b, n, tf, f):
    # tf (bool): Variable to decide whether to perform Error evaluation

    # calculating step size
    h = (b - a) / n
    if tf:
        print(bcolors.FAIL, "Error evaluation En = ", round(TrapezError(func(), b, a, h), 6), bcolors.ENDC)

    # Finding sum
    integration = f(a) + f(b)

    for i in range(1, n):
        k = a + i * h
        integration = integration + 2 * f(k)

    # Finding final integration value
    integration = integration * h / 2

    return integration
def TrapezError(func, b, a, h):
    """
    The trapezoidal rule is a method for approximating definite integrals of functions.
    The error in approximating the integral of a twice-differentiable function by the trapezoidal rule
    is proportional to the second derivative of the function at some point in the interval.
    :param func: The desired integral function
    :param b: Upper bound
    :param a: Lower bound
    :param h: The division
    :return: The error value
    """
    xsi = (-1) * (math.pi / 2)
    print("ƒ(x): ", func)
    f2 = sp.diff(func, x, 2)
    print("ƒ'': ", f2)
    diff_2 = lambdify(x, f2)
    print("ƒ''(", xsi, ") =", diff_2(xsi))
    return h**2 / 12 * (b-a) * diff_2(xsi)


def SimpsonError(func, b, a, h):
    """
    The Simpson rule is a method for approximating definite integrals of functions.
    The error in approximating the integral of a four-differentiable function by the trapezoidal rule
    is proportional to the second derivative of the function at some point in the interval.
    :param func: The desired integral function
    :param b: Upper bound
    :param a: Lower bound
    :param h: The division
    :return: The error value
    """
    xsi = 1
    print("ƒ(x): ", func)
    f2 = sp.diff(func, x, 4)
    print("ƒ⁴: ", f2)
    diff_4 = sp.lambdify(x, f2)
    print("ƒ⁴(", xsi, ") =", diff_4(xsi))

    return (math.pow(h, 4) / 180)*(b-a)*diff_4(xsi)
def func():
    return sp.sin(x)

def f(val):
    return lambdify(x, func())(val)

if __name__ == "__main__":


    f = lambda x: sp.sin(x)
    n = 4
    a = 0
    b = math.pi
    epsilon=0.0000001
    print(bcolors.BOLD, "Division into sections n =", n, bcolors.ENDC)
    print(bcolors.OKBLUE, "Numerical Integration of definite integral in range [0,𝛑] ∫= SIN(X)", bcolors.ENDC)
    choice = int(input(
        "Which method do you want? \n\t1.The Trapezoidal Rule \n\t2.Simpson’s Rule\n\t3.Romberg's method\n"))
    if choice == 1:
        result = trapezoidal(a, b, n, True, f)
        print(bcolors.OKBLUE, "I = ", round(result, 6), bcolors.ENDC)

    elif choice == 2:
        integral  = simpsons_rule(f, a, b,  n,True)
        if integral :
            print(bcolors.OKBLUE, "I = ", round(integral,6), bcolors.ENDC)
    elif choice == 3:
        #integral = my_romberg_integration(f, a, b, n)
        integral = romberg_integration1(f, a, b, n, epsilon)
        print(bcolors.OKBLUE, "I = ", round(integral, 5), bcolors.ENDC)
    else:
        print(bcolors.FAIL, "Invalid input", bcolors.ENDC)




