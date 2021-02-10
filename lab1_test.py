import numpy as np
import scipy.optimize as optimizers
from scipy.optimize import LinearConstraint


def check_condition(na, nb):
    if da * na + db * nb <= 1700 \
            and 6000 <= ta * na + tb * nb <= 9600 \
            and na >= 0 and nb >= 0:
        return True
    else:
        return False


def get_results(na, nb):
    return y1((na, nb)), y2((na, nb))


if __name__ == '__main__':
    pa = 2
    pb = 4

    da = 3
    db = 4

    ta = 12
    tb = 30

    y1 = lambda N: pa * N[0] + pb * N[1]  # -> min profit
    y2 = lambda N: 0 if all([n == 0 for n in N]) else y1(N) / (ta * N[0] + tb * N[1])  # -> min unit profit

    z1 = lambda N: -y1(N)  # -> max profit
    z2 = lambda N: -y2(N)  # -> max unit profit

    bounds = (
        (0, np.inf),
        (0, np.inf)
    )

    constraints = [
        LinearConstraint(A=[da, db], lb=0, ub=1700),
        LinearConstraint(A=[ta, tb], lb=6000, ub=9600),
        LinearConstraint(A=[1, 0], lb=bounds[0][0], ub=bounds[0][1], keep_feasible=True),
        LinearConstraint(A=[0, 1], lb=bounds[1][0], ub=bounds[1][1], keep_feasible=True)
    ]

    print_res = lambda res: print(f"best_na = {res.x[0]}, best_nb = {res.x[1]}\nbest_fun = {-res.fun}\n")

    res2 = optimizers.minimize(
        z1,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    print_res(res2)

    res2 = optimizers.minimize(
        z2,
        x0=np.array([bound[0] for bound in bounds]),
        constraints=constraints,
        method='trust-constr'
    )
    print_res(res2)
