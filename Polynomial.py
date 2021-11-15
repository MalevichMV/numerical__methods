import numpy as np
import sympy as sp


def jacobi_matrix(values, n, function):
    matrix = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = df(function[i], values, j + 1, n)
    return matrix


def descent(n, function, e):
    eps = e
    x_prev = [i for i in range(n)]
    while True:
        matrix = jacobi_matrix(x_prev, n, function)
        func = []
        for i in range(n):
            func.append(f(sp.parse_expr(function[i]), x_prev, n))

        g = np.dot(np.transpose(matrix), func)
        lam = ((np.dot(np.transpose(g), g)) / (
            np.dot(np.dot(np.dot(np.transpose(g), np.transpose(matrix)), matrix), g))) * 0.5

        x_new = x_prev - lam * (2 * np.dot(np.transpose(matrix), func))
        if np.linalg.norm(x_new - x_prev) < eps:
            return abs(x_new)
        x_prev = x_new


def iteration_method(n, function):
    eps = 0.00001
    x_prev = [i for i in range(n)]
    matrix = jacobi_matrix(x_prev, n, function)
    reverse_matrix = np.linalg.inv(matrix)
    while True:
        func = []
        for i in range(n):
            func.append(f(sp.parse_expr(function[i]), x_prev, n))

        x_new = x_prev - np.dot(reverse_matrix, func)
        if np.linalg.norm(x_new - x_prev) < eps:
            return x_new
        x_prev = x_new


def f(func, value, n):
    temp = []
    for i in range(n):
        temp.append(sp.symbols(f't{i + 1}'))
        func = func.subs([(temp[i], value[i])])
    return float(func)


def df(func, values, current_arg, n):
    x = sp.symbols(f't{current_arg}')
    pol_str = sp.diff(sp.parse_expr(func), x)
    return f(pol_str, values, n)