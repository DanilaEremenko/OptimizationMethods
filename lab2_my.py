import numpy as np

from lab_2_finit import get_finit_table_list

if __name__ == '__main__':
    # стратегии
    g = [
        "с рекламой",
        "без рекламы"
    ]

    # состоянияф
    state_list = [
        "okay",
        "not okay"
    ]

    table_list = get_finit_table_list(
        N=3,
        g_list=g,
        state_list=state_list,
        p_list=[
            # матрица вероятностности с реклмой
            np.array([
                [0.9, 0.1],
                [0.6, 0.4]
            ]),
            # матрица вероятностности без рекламы
            np.array([
                [0.7, 0.3],
                [0.2, 0.8]
            ])
        ],
        r_list=[
            # матрица прибыли с реклмой
            np.array([
                [2, -1],
                [1, -3]
            ]),
            # матрица прибыли без рекламы
            np.array([
                [4, 1],
                [2, -1]
            ])
        ],
        a=0.9
    )
