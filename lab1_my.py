"""
На предприятии по производству детских игрушек стоит цель повысить производительность за отчетный период.

Для этого сотрудникам оплачены сверхурочные часы – 600 часов.
На нем производятся следующие товары:
Мягкие игрушки, календари, конструкторы, детская железная дорога, пластмассовые игрушки.
Известно, для того чтобы произвести каждый из перечисленных товаров нужно потратить 4, 2, 7, 8, 3 сверхурочных часа
соответственно, при этом их розничная цена соответственно равна – 15$, 5$, 27$, 35$, 7 $ .

Работникам сообщили, что нужно дополнительно (за счет сверхурочной работы)
произвести как минимум по 3 единицы каждого товара,
при этом комплектующих, оставшихся на заводе хватит только на производство
-30 железных дорог и
-4 наборов конструкторов,

-а спрос на  мягкие игрушки в 2 раза выше, чем на железную дорогу (дополнительно минимизация разницы?)

Торгующая организация формирует подарочные комплекты, состоящие из
-3-х мягких игрушек
-1 календаря,
-2 конструкторов,
-1 железной дороги
-1 пластмассовой игрушки.

Требуется максимизировать
-полученную выручку
-количество комплектов,
-при этом использовать все выделенные сверхурочные часы по максимуму.
"""

import scipy.optimize as optimizers
import numpy as np
from scipy.optimize import LinearConstraint


def check_condition(args):
    if time_lb <= sum([n * t for n, t in zip(args, t_arr)]) <= time_ub:  # TODO add statements
        return True
    else:
        return False


def get_results(args):
    return y1(args), y2(args)


def y1(N):  # выручка
    return sum([n * p for n, p in zip(N, p_arr)])


def y2(N):  # количество комплектов
    """
    Торгующая организация формирует подарочные комплекты, состоящие из
    -3-х мягких игрушек
    -1 календаря,
    -2 конструкторов,
    -1 железной дороги
    -1 пластмассовой игрушки.
    """
    return min([n // set_need for n, set_need in zip(N, set_arr)])


def verbose_res_print(res, title):
    row_args = tuple(res.x)
    actual_args = tuple(int(round(arg, 0)) for arg in res.x)

    print(f"---{title.upper()}---")
    print(f"row args        : {', '.join([f'{name} = %.2f' for name in n_arr])}" % (row_args))
    print(f"actual args     : {', '.join([f'{name} = %.2f' for name in n_arr])}" % (actual_args))
    # print(f"fun = {-res.fun}")

    cond_respected = check_condition(actual_args)

    print(f"conditions respected: {cond_respected}")

    if cond_respected:
        # print(f"\td remeains: {d_max - da * actual_args[0] - db * actual_args[1]}")
        # print(f"\tt remains: {t_max - ta * actual_args[0] - tb * actual_args[1]}")
        print("y1 = %.2f, y2 = %.2f\n" % (get_results(actual_args)))


if __name__ == '__main__':
    n_arr = ['soft toys', 'calendar', 'constructor', 'railway road', 'plastic toys']

    t_arr = [4, 2, 7, 8, 3]  # массив необходимого фвремени
    p_arr = [15, 5, 27, 35, 7]  # массив цен

    set_arr = [3, 1, 2, 1, 1]  # массив количества необходимо для одного набора

    x0 = np.array([0, 0, 0, 0, 0])  # начальные состояния

    time_lb = 0
    time_ub = 600

    z1 = lambda N: -y1(N)  # -> максимизация прибыли
    z2 = lambda N: -y2(N)  # -> максимизация наборов

    ######################################################################
    # ----------------------- ОГРАНИЧЕНИЯ --------------------------------
    ######################################################################
    # время(N) < 600
    time_constraints = [
        LinearConstraint(
            A=t_arr,
            lb=time_lb,
            ub=time_ub
        ),
    ]

    # цена(N) < цены(30 железныз дорог и 4 наборов)
    material_constrains = [
        LinearConstraint(
            A=p_arr,
            lb=0,
            ub=30 * p_arr[3] + 4 * sum([p * set_n for p, set_n in zip(p_arr, set_arr)])
        )
    ]

    # количество каждой игрушки >= 3
    amount_constraints = [
        LinearConstraint(A=[1, 0, 0, 0, 0], lb=3, ub=np.inf),
        LinearConstraint(A=[0, 1, 0, 0, 0], lb=3, ub=np.inf),
        LinearConstraint(A=[0, 0, 1, 0, 0], lb=3, ub=np.inf),
        LinearConstraint(A=[0, 0, 0, 1, 0], lb=3, ub=np.inf),
        LinearConstraint(A=[0, 0, 0, 0, 1], lb=3, ub=np.inf),
    ]

    all_constraints = [
        *time_constraints,
        *material_constrains,
        *amount_constraints
    ]

    ######################################################################
    # ----------------------- РЕШЕНИЕ ------------------------------------
    ######################################################################
    print_res = lambda res: print(', '.join([f"{name} = {n}" for name, n in zip(n_arr, res.x)]))

    res1 = optimizers.minimize(
        z1,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr'

    )
    verbose_res_print(res1, title="z1 separate")

    res2 = optimizers.minimize(
        z2,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr'
    )
    verbose_res_print(res2, title="z2 separate")
