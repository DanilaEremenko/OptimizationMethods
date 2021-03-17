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
from scipy.optimize import LinearConstraint
import numpy as np


def check_condition(N):
    if time_lb <= sum([n * t for n, t in zip(N, t_arr)]) <= time_ub \
            and mat_lb <= sum([p * n for p, n in zip(p_arr, N)]) <= mat_ub \
            and all([n >= 3 for n in N]):  # TODO add statements
        return True
    else:
        return False


def get_results(args):
    return y1(args), int(y2(args))


def verbose_res_print(N, title):
    row_args = tuple(N)
    actual_args = tuple(int(arg) for arg in N)

    print(f"---{title.upper()}---")
    print(f"row args        : {', '.join([f'{name} = %.2f' for name in n_arr])}" % (row_args))
    print(f"actual args     : {', '.join([f'{name} = %.2f' for name in n_arr])}" % (actual_args))

    cond_respected = check_condition(row_args)

    print(f"conditions respected: {cond_respected}")

    print(f"\tt row remains: {time_ub - sum([t * n for t, n in zip(t_arr, row_args)])}")
    print(f"\tm row remains: {mat_ub - sum([p * n for p, n in zip(p_arr, row_args)])}")
    print(f"\tt actual remains: {time_ub - sum([t * n for t, n in zip(t_arr, actual_args)])}")
    print(f"\tm actual remains: {mat_ub - sum([p * n for p, n in zip(p_arr, actual_args)])}")
    if cond_respected:
        print("row y1 = %.2f, y2 = %.2f" % (get_results(row_args)))
        print("actual y1 = %.2f, y2 = %.2f\n" % (get_results(actual_args)))


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
    return min([n / set_need for n, set_need in zip(N, set_arr)])


if __name__ == '__main__':
    n_arr = ['soft toys', 'calendar', 'constructor', 'railway road', 'plastic toys']

    t_arr = [4, 2, 7, 8, 3]  # массив необходимого времени
    p_arr = [15, 5, 27, 35, 7]  # массив цен
    benefit_arr = [p / t for p, t in zip(p_arr, t_arr)]  # массив коэффицентов выгоды

    set_arr = [3, 1, 2, 1, 1]  # массив количества необходимо для одного набора

    x0 = np.array([3, 3, 3, 3, 3])  # начальные состояния

    z1 = lambda N: -y1(N)  # -> максимизация прибыли
    z2 = lambda N: -y2(N)  # -> максимизация наборов

    ######################################################################
    # ----------------------- ОГРАНИЧЕНИЯ --------------------------------
    ######################################################################
    # 0 <= время(N) <= 600
    time_lb = 0.0
    time_ub = 600.0
    time_constraints = LinearConstraint(
        A=t_arr,
        lb=time_lb,
        ub=time_ub
    )

    # 0 <= цена(N) <= цены(30 железныз дорог и 4 наборов)
    mat_lb = 0.0
    mat_ub = 30 * p_arr[3] + 4 * sum([p * set_n for p, set_n in zip(p_arr, set_arr)])
    material_constrains = LinearConstraint(
        A=p_arr,
        lb=0,
        ub=mat_ub
    )

    # спрос на  мягкие игрушки в 2 раза выше, чем на железную дорогу
    # x1/x4>=0
    demand_constraints = LinearConstraint(
        A=[-1, 0, 0, 2, 0],
        lb=0,
        ub=0
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
        demand_constraints,
        amount_constraints,
    ]

    # hess = lambda x: np.zeros(len(x))
    hess = None

    ######################################################################
    # ----------------------- РЕШЕНИЕ ------------------------------------
    ######################################################################
    res_y1_indep = optimizers.minimize(
        z1,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_y1_indep.x, title="z1 separate")

    res_y2_indep = optimizers.minimize(
        z2,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_y2_indep.x, title="z2 separate")

    ######################################################################
    # ----------------------- ВЫДЕЛЕНИЕ ГЛАВНОГО КРИТЕРИЯ ----------------
    ######################################################################
    all_constraints_with_main = [
        *all_constraints,
        LinearConstraint(
            A=p_arr,
            lb=0,
            ub=(-res_y1_indep.fun) * 0.9
        )
    ]

    res_y1_main_crit = optimizers.minimize(
        z2,
        x0=x0,
        constraints=all_constraints_with_main,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_y1_main_crit.x, title="z2 with selecting z1 as the main criterion")

    ######################################################################
    # ----------------------- АДДИТИВНАЯ СВЕРТКА -------------------------
    ######################################################################
    z_addit = lambda N: 0.5 * z1(N) / (-res_y1_indep.fun) + 0.5 * z2(N) / (-res_y2_indep.fun)

    res_addit_conv = optimizers.minimize(
        z_addit,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_addit_conv.x, title="z additive convolution")

    ######################################################################
    # ----------------------- МУЛЬТИПЛИКАТИВНАЯ СВЕРТКА ------------------
    ######################################################################
    z_mult = lambda N: 1 / (z1(N) / (-res_y1_indep.fun) * z2(N) / (-res_y2_indep.fun))

    res_mult_conv = optimizers.minimize(
        z_mult,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_mult_conv.x, title="z multiplicative convolution")

    ######################################################################
    # ----------------------- ПОСЛЕДОВАТЕЛЬНЫЕ УСТУПКИ ------------------
    ######################################################################
    all_constraints_for_successive_assignments = [
        *all_constraints,
        LinearConstraint(
            A=p_arr,
            lb=-res_y1_indep.fun * 0.95,
            ub=np.inf
        )
    ]

    res_succ_assgm = optimizers.minimize(
        z2,
        x0=x0,
        constraints=all_constraints_for_successive_assignments,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_succ_assgm.x, title="z2 with z1 in constraints (successive assignments)")

    ######################################################################
    # ----------------ВВЕДЕНИЕ МЕТРИКИ В ПРОСТРАНСТВЕ КРИТЕРИЕВ ----------
    ######################################################################
    z_metric = lambda N: -(1 - (z1(N) / -res_y1_indep.fun)) ** 2 + (1 - (z2(N) / -res_y2_indep.fun)) ** 2

    res_metric = optimizers.minimize(
        z_metric,
        x0=x0,
        constraints=all_constraints,
        method='trust-constr',
        hess=hess
    )
    verbose_res_print(res_metric.x, title="z metric")
