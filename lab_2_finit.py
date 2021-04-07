import numpy as np
import pandas as pd


def get_finit_table_list(N, g_list: list, state_list: list, p_list: list, r_list: list, a=1):
    k_len = len(g_list)
    j_len = len(state_list)
    table_list = [np.zeros(shape=(j_len, k_len))]

    # FIRST
    print(f"---------")
    print(f"N = {0}")
    table_list.append(np.zeros(shape=(j_len, k_len)))
    for j in range(j_len):
        for k in range(k_len):
            table_list[-1][j, k] = a * sum(p_list[k][j] * r_list[k][j])
    print(f"TABLE\n{table_list[-1]}")

    # OTHER
    for n in range(1, N):
        print(f"---------")
        print(f"N = {n}")
        r_prev = [line.max() for line in table_list[-1]]
        table_list.append(np.zeros(shape=(j_len, k_len)))
        for j in range(j_len):
            for k in range(k_len):
                table_list[-1][j, k] = table_list[1][j][k] + a * sum(p_list[k][j] * r_prev)
        print(f"TABLE\n{table_list[-1]}")

    return table_list
