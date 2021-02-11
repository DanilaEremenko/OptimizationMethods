import numpy as np
import scipy.optimize as optimizers
from scipy.optimize import LinearConstraint


def check_condition(na, nb):
    if d_min <= da * na + db * nb <= d_max \
            and t_min <= ta * na + tb * nb <= t_max \
            and na >= 0 \
            and nb >= 0:
        return True
    else:
        return False


def get_results(na, nb):
    return y1((na, nb)), y2((na, nb))


def verbose_res_print(res, title):
    actual_results = get_results(res.x[0], res.x[1])
    print(f"---{title.upper()}---")
    print(f"na = {res.x[0]}, nb = {res.x[1]}")
    # print(f"fun = {-res.fun}")
    print(f"conditions respected: {check_condition(res.x[0], res.x[1])}")
    print(f"y1 = {actual_results[0]}, y2 = {actual_results[1]}\n")


if __name__ == '__main__':
    # parameters
    pa = 2
    pb = 4

    da = 3
    db = 4

    ta = 12
    tb = 30

    # target functions
    y1 = lambda N: pa * N[0] + pb * N[1]  # -> min profit
    y2 = lambda N: 0 if all([n == 0 for n in N]) else y1(N) / (ta * N[0] + tb * N[1])  # -> min unit profit

    z1 = lambda N: -y1(N)  # -> max profit
    z2 = lambda N: -y2(N)  # -> max unit profit

    # bounds & constrains
    bounds = (
        (0, np.inf),
        (0, np.inf)
    )

    d_min = 0
    d_max = 1700

    t_min = 6000
    t_max = 9600

    constraints = [
        LinearConstraint(A=[da, db], lb=d_min, ub=d_max),
        LinearConstraint(A=[ta, tb], lb=t_min, ub=t_max),
        LinearConstraint(A=[1, 0], lb=bounds[0][0], ub=bounds[0][1], keep_feasible=True),
        LinearConstraint(A=[0, 1], lb=bounds[1][0], ub=bounds[1][1], keep_feasible=True)
    ]

    # --------- CALCULATE Y1, Y2 SEPARATE --------------
    z1_opt = optimizers.minimize(
        z1,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    verbose_res_print(z1_opt, title="z1 separate")

    z2_opt = optimizers.minimize(
        z2,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    verbose_res_print(z2_opt, "z2 separate")

    # ------------- ADDITIVE CONVOLUTION ---------------
    z1_norm = lambda N: z1(N) / abs(z1_opt.fun)
    z2_norm = lambda N: z2(N) / abs(z2_opt.fun)

    additive_fun = lambda N: 0.5 * z1_norm(N) + 0.5 * z2_norm(N)

    additive_res = optimizers.minimize(
        additive_fun,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    verbose_res_print(additive_res, "additive")

    # ----------- MULTIPLICATIVE CONVOLUTION -----------
    multiplicative_fun = lambda N: 1 / (z1(N) * z2(N)) if z1(N) != 0 and z2(N) != 0 else 1000

    multiplicative_res = optimizers.minimize(
        multiplicative_fun,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    verbose_res_print(multiplicative_res, "multiplicative")

    # --------------- MINIMAX --------------------------
    worse_y2_fun = lambda N: -z2(N) / (-z2_opt.fun)
    best_y1_fun = lambda N: z1(N) / (-z1_opt.fun)

    max_res = optimizers.minimize(
        worse_y2_fun,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    verbose_res_print(max_res, "worse y2 in minimax")

    min_res = optimizers.minimize(
        best_y1_fun,
        x0=max_res.x,
        constraints=constraints,
        method='trust-constr'
    )
    verbose_res_print(min_res, "best y1 in minimax")
