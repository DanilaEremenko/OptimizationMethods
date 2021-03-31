import numpy as np


def brood_force(curr_state: int, strat_list: list, state_list: list, p_list: list, r_list: list):
    m = len(state_list)
    Ek_list = []
    for k, (pk_matrix, rk_matrix) in enumerate(zip(p_list, r_list)):
        # 1.Вычисление ожидаемого дохода за один шаг при k-ой стационарной стратегии
        vi = [pk_matrix[curr_state][n] * rk_matrix[curr_state][n] for n in range(m)]

        # 2.Вычисление матрицы стационарных вероятностей
        Im = np.identity(len(strat_list))
        stnry_vector = np.linalg.solve(a=pk_matrix - Im, b=[0] * m)

        # 3.Определение ожидаемого дохода для всех стационарных стратегий
        Ek_list.append([curr_stnry * curr_vi for curr_stnry, curr_vi in zip(stnry_vector, vi)])

    return max(range(len(Ek_list)), key=Ek_list.__getitem__)


if __name__ == '__main__':
    # стратегии
    strat_list = [
        "с рекламой",
        "без рекламы"
    ]

    state_list = [
        "okay",
        "not okay"
    ]

    # матрица вероятностности с реклмой
    P1 = np.array([
        [0.9, 0.1],
        [0.6, 0.4]
    ])

    # матрица вероятностности без рекламы
    P2 = np.array([
        [0.7, 0.3],
        [0.2, 0.8]
    ])

    # матрица прибыли с реклмой
    R1 = np.array([
        [2, -1],
        [1, -3]
    ])

    # матрица прибыли без рекламы
    R2 = np.array([
        [4, 1],
        [2, -1]
    ])
    step_num = 5
    for curr_step in range(step_num):
        curr_state = np.random.randint(0, len(state_list))
        best_strategy_i = brood_force(
            curr_state=curr_state,
            strat_list=strat_list,
            state_list=state_list,
            p_list=[P1, P2],
            r_list=[R1, R2]
        )
        print(f"{curr_step}: best_strategy = '{strat_list[best_strategy_i]}', if state = {curr_state}")
