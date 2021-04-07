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


if __name__ == '__main__':
    table_list = get_finit_table_list(
        N=3,
        g_list=['X1', 'X2'],
        state_list=['s1', 's2', 's3'],
        p_list=[
            np.array([
                [0.2, 0.5, 0.3],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0]]),
            np.array([
                [0.3, 0.6, 0.1],
                [0.1, 0.6, 0.3],
                [0.05, 0.4, 0.55]])
        ],
        r_list=[
            np.array([
                [7, 6, 3],
                [0, 5, 1],
                [0, 0, -1]]),
            np.array([
                [6, 5, -1],
                [7, 4, 0],
                [6, 3, -2]])
        ],
        a=1
    )
