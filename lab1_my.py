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
-а спрос на  мягкие игрушки в 2 раза выше, чем на железную дорогу

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


def check_condition(N):
    if time_lb <= sum([n * t for n, t in zip(N, t_arr)]) <= time_ub \
            and all([n >= 3 for n in N]):  # TODO add statements
        return True
    else:
        return False


def get_results(args):
    return y1(args), y2(args)


def verbose_res_print(N, title):
    row_args = tuple(N)
    actual_args = tuple(int(round(arg, 0)) for arg in N)

    print(f"---{title.upper()}---")
    print(f"row args        : {', '.join([f'{name} = %.2f' for name in n_arr])}" % (row_args))
    print(f"actual args     : {', '.join([f'{name} = %.2f' for name in n_arr])}" % (actual_args))

    cond_respected = check_condition(actual_args)

    print(f"conditions respected: {cond_respected}")

    if cond_respected:
        print(f"\tt remains: {time_ub - sum([t * n for t, n in zip(t_arr, actual_args)])}")
        print("y1 = %.2f, y2 = %.2f\n" % (get_results(actual_args)))


def y1(N):  # выручка
    assert len(N) == len(p_arr) == 5
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
    assert len(N) == len(set_arr) == 5
    return min([n // set_need for n, set_need in zip(N, set_arr)])


if __name__ == '__main__':
    n_arr = ['soft toys', 'calendar', 'constructor', 'railway road', 'plastic toys']

    t_arr = [4, 2, 7, 8, 3]  # массив необходимого фвремени
    p_arr = [15, 5, 27, 35, 7]  # массив цен

    set_arr = [3, 1, 2, 1, 1]  # массив количества необходимо для одного набора

    x0 = np.array([3, 3, 3, 3, 3])  # начальные состояния

    time_lb = 0.0
    time_ub = 600.0

    z1 = lambda N: -y1(N)  # -> максимизация прибыли
    z2 = lambda N: -y2(N)  # -> максимизация наборов

    ######################################################################
    # ----------------------- ОГРАНИЧЕНИЯ --------------------------------
    ######################################################################
    # время(N) < 600
    time_constraints = LinearConstraint(
        A=t_arr,
        lb=time_lb,
        ub=time_ub
    )

    # цена(N) < цены(30 железныз дорог и 4 наборов)
    material_constrains = LinearConstraint(
        A=p_arr,
        lb=0,
        ub=30 * p_arr[3] + 4 * sum([p * set_n for p, set_n in zip(p_arr, set_arr)])
    )

    # спрос на  мягкие игрушки в 2 раза выше, чем на железную дорогу
    demand_constraints = LinearConstraint(
        A=[2, 0, 0, 1, 0],
        lb=0,
        ub=0,
        keep_feasible=True
    )

    # количество каждой игрушки >= 3
    amount_constraints = LinearConstraint(
        A=[[1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1]],
        lb=[3, 3, 3, 3, 3],
        ub=[np.inf, np.inf, np.inf, np.inf, np.inf]
    )

    all_constraints = [
        time_constraints,
        material_constrains,
        amount_constraints,
        demand_constraints,
    ]

    ######################################################################
    # ----------------------- РЕШЕНИЕ ------------------------------------
    ######################################################################
    res1 = optimizers.minimize(
        z1,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr'

    )
    verbose_res_print(res1.x, title="z1 separate")

    res2 = optimizers.minimize(
        z2,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr'
    )
    verbose_res_print(res2.x, title="z2 separate")
