import numpy as np
from fractions import Fraction


def print_table_info(vi, stnry_vector):
    for i, curr_v in enumerate(vi):
        print(f"v{i + 1}k = {curr_v}")

    print()
    for i, curr_strny in enumerate(stnry_vector):
        print(f"П{i + 1}k = {round(curr_strny, 2)}")


def brood_force(strat_list: list, state_list: list, p_list: list, r_list: list):
    m = len(state_list)
    Ek_list = []
    for k, (pk_matrix, rk_matrix) in enumerate(zip(p_list, r_list)):
        # 1.Вычисление ожидаемого дохода за один шаг при k-ой стационарной стратегии
        vi = [(pk_matrix[n_from, :] * rk_matrix[n_from, :]).sum() for n_from in range(m)]

        # 2.Вычисление матрицы стационарных вероятностей
        Im = np.identity(m)
        det = np.linalg.det(pk_matrix.transpose() - Im)

        print("-------------")
        print(f"k = {k + 1}")
        print(f"det = {det}")
        if det == 0:
            a = np.append(pk_matrix.transpose() - Im, np.array([1] * len(state_list))) \
                .reshape(len(strat_list), len(state_list))
            b = [*[0] * m, 1]
            solve_res = np.linalg.lstsq(a=a, b=b, rcond=None)
            stnry_vector = solve_res[0]

            print_table_info(vi=vi, stnry_vector=stnry_vector)

            # 3.Определение ожидаемого дохода для всех стационарных стратегий
            Ek_list.append(round(sum([curr_stnry * curr_vi for curr_stnry, curr_vi in zip(stnry_vector, vi)]), 2))
        else:
            Ek_list.append(-1)
    # 4. Определение номера k оптимальной стационарной стратегии
    print(f"Ek = {Ek_list}\n")
    return max(range(len(Ek_list)), key=Ek_list.__getitem__)


if __name__ == '__main__':
    # стратегии
    strat_list = [
        "strat1",
        "strat2",
        "strat3",
        "strat4"
    ]

    state_list = [
        "s1",
        "s2",
        "s3"
    ]

    P1 = np.array([
        [0.2, 0.5, 0.3],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, -1.0]
    ])

    P2 = np.array([
        [0.3, 0.6, 0.1],
        [0.1, 0.6, 0.3],
        [0.05, 0.4, 0.55]
    ])

    P3 = np.array([
        [0.3, 0.6, 0.1],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0]
    ])

    P4 = np.array([
        [0.2, 0.5, 0.3],
        [0.1, 0.6, 0.3],
        [0.0, 0.0, 1.0]
    ])

    R1 = np.array([
        [7, 6, 3],
        [0, 5, 1],
        [0, 0, 1]
    ])

    R2 = np.array([
        [6, 5, -1],
        [7, 4, 0],
        [6, 3, -2]
    ])

    R3 = np.array([
        [6, 5, -1],
        [0, 5, 1],
        [0, 0, -1]
    ])

    R4 = np.array([
        [7, 6, 3],
        [7, 4, 0],
        [0, 0, -1]
    ])

    step_num = 1
    for curr_step in range(step_num):
        curr_state = np.random.randint(0, len(state_list))
        best_strategy_i = brood_force(
            strat_list=strat_list,
            state_list=state_list,
            p_list=[P1, P2, P3, P4],
            r_list=[R1, R2, R3, R4]
        )
        print(f"{curr_step}: best_strategy = '{strat_list[best_strategy_i]}', if state = {curr_state}")
