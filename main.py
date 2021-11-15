import numpy as np
import sympy as sp
import Polynomial as p


def clear_output():
    f = open(f'Output.txt', 'w')
    f.close()


class Integral:
    @staticmethod
    def read_from_file():
        f = open(f'Input.txt', 'r')
        temp_list = [line.strip() for line in f]
        f.close()
        return temp_list

    @staticmethod
    def write_to_file(*args):
        f = open(f'Output.txt', 'a')
        for item in args:
            f.write(str(item) + "\n" + "\n")

    def __init__(self):
        data_list = Integral.read_from_file()

        self.task_type = int(data_list[0])
        self.grid_type = data_list[1]
        self.n = int(data_list[2])

        if self.grid_type == "even" or self.task_type == 5:
            self.bounds = np.asarray([item for item in data_list[3].split()], dtype=np.float32)
            self.h = (self.bounds[1] - self.bounds[0]) / self.n

        if self.grid_type == "uneven":
            self.x = np.asarray([item for item in data_list[3].split()], dtype=np.float32)

        # if self.grid_type != "dynamic" and self.task_type != 5:
        if data_list[4] == "table":
            self.y = np.asarray([item for item in data_list[5].split()], dtype=np.float32)

        if data_list[4] == "analytical":
            self.func = data_list[5]
            if self.grid_type == "uneven":
                self.y = [self.func_f(item) for item in self.x]

            if self.grid_type == "even":
                x = self.bounds[0]
                self.y = []
                while x <= self.bounds[1]:
                    self.y.append(self.func_f(x))
                    x += self.h

        if self.grid_type == "dynamic" or self.task_type == 5:
            self.eps = float(data_list[6])

    def func_f(self, value):
        pol_str = sp.parse_expr(self.func)
        x = sp.symbols('x')
        return pol_str.subs(x, value)

    def left_rect(self):
        if self.grid_type == "uneven":
            result = 0
            for i in range(self.n):
                result += self.y[i] * (self.x[i + 1] - self.x[i])
            self.write_to_file(result)

        if self.grid_type == "even":
            self.write_to_file(self.h * (sum(self.y) - self.y[self.n]))

    def right_rect(self):
        if self.grid_type == "uneven":
            result = 0
            for i in range(self.n):
                result += self.y[i + 1] * (self.x[i + 1] - self.x[i])
            self.write_to_file(result)

        if self.grid_type == "even":
            self.write_to_file(self.h * (sum(self.y) - self.y[0]))

    def trapezoid_method(self):
        if self.grid_type == "uneven":
            result = 0
            for i in range(self.n):
                result += (self.x[i + 1] - self.x[i]) * (self.y[i] + self.y[i + 1])
            self.write_to_file(result * 0.5)

        if self.grid_type == "even":
            result = self.y[0] + 2 * (sum(self.y) - self.y[0] - self.y[self.n]) + self.y[self.n]
            self.write_to_file(result * (self.h * 0.5))

    def simpsone_method(self):
        n = int(self.n * 0.5)

        if self.grid_type == "uneven":
            result = 0
            for i in range(n):
                h_r = self.x[2 * i + 1] - self.x[2 * i]
                h_r1 = self.x[2 * i + 2] - self.x[2 * i + 1]
                result += ((h_r1 + h_r) / (6 * h_r1 * h_r)) * (
                        h_r1 * (2 * h_r - h_r1) * self.y[2 * i] + (h_r1 + h_r) ** 2 * self.y[2 * i + 1] + h_r * (
                        2 * h_r1 - h_r) * self.y[2 * i + 2])
            self.write_to_file(result)

        if self.grid_type == "even":
            summ1 = summ2 = 0
            for i in range(1, n):
                summ1 += self.y[2 * i + 1]
                summ2 = self.y[2 * i]
            self.write_to_file((2 * self.h / 6) * (self.y[0] + 4 * (summ1 + self.y[1]) + 2 * summ2 + self.y[2 * n]))

    def chebyshev_method(self):
        functions_list = []
        for i in range(1, self.n + 1):
            temp_str = ""
            temp = (self.n * (1 + (-1) ** i)) / (2 * (i + 1))
            for j in range(1, self.n + 1):
                temp_str += f't{j}**{i}+'
            functions_list.append(temp_str + f"(-{temp})")

        t = p.descent(self.n, functions_list, self.eps)

        result = 0
        error_list = []
        for i in range(self.n):
            error_list.append(self.func_f(
                (self.bounds[1] + self.bounds[0]) * 0.5 + ((self.bounds[1] - self.bounds[0]) * 0.5) * t[i]))
            result += error_list[i]

        self.write_to_file((self.bounds[1] - self.bounds[0]) / self.n * result)


clear_output()
integral = Integral()
if integral.task_type == 1:
    integral.left_rect()
elif integral.task_type == 2:
    integral.right_rect()
elif integral.task_type == 3:
    integral.trapezoid_method()
elif integral.task_type == 4:
    integral.simpsone_method()
elif integral.task_type == 5:
    integral.chebyshev_method()
else:
    integral.write_to_file('Incorrect number!')
