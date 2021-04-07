import numpy as np


def print_brood_force(vi, stnry_vector):
    for i, curr_v in enumerate(vi):
        print(f"v{i + 1}k = {round(curr_v, 2)}")

    print()
    for i, curr_strny in enumerate(stnry_vector):
        print(f"П{i + 1}k = {round(curr_strny, 2)}")


def brood_force(state_list: list, p_list: list, r_list: list):
    m = len(state_list)
    Ek_list = []
    Im = np.identity(m)

    for k, (pk_matrix, rk_matrix) in enumerate(zip(p_list, r_list)):
        print("-------------")
        print(f"k = {k + 1}")

        # 1.Вычисление ожидаемого дохода за один шаг при k-ой стационарной стратегии
        vi = [(pk_matrix[n_from, :] * rk_matrix[n_from, :]).sum() for n_from in range(m)]

        # 2.Вычисление матрицы стационарных вероятностей
        det = np.linalg.det(pk_matrix.transpose() - Im)
        det = round(det, 0)
        print(f"det = {det}")
        if det == 0.0:
            a = np.append(pk_matrix.transpose() - Im, np.array([1] * len(state_list))) \
                .reshape(len(state_list) + 1, len(state_list))
            b = [*[0] * m, 1]
            solve_res = np.linalg.lstsq(a=a, b=b, rcond=None)
            stnry_vector = solve_res[0]

            print_brood_force(vi=vi, stnry_vector=stnry_vector)

            # 3.Определение ожидаемого дохода для всех стационарных стратегий
            Ek_list.append(round(sum([curr_stnry * curr_vi for curr_stnry, curr_vi in zip(stnry_vector, vi)]), 2))
        else:
            Ek_list.append(-1.0)
    # 4. Определение номера k оптимальной стационарной стратегии
    print(f"Ek = {Ek_list}\n")
    return max(range(len(Ek_list)), key=Ek_list.__getitem__)


def print_iter_by_strat(vi, stnry_vector):
    for i, curr_v in enumerate(vi):
        print(f"v{i + 1}k = {round(curr_v, 2)}")

    print()
    print("Er = %.2f, Fr(1) = %.2f, Fr(2) = %.2f" % (tuple(stnry_vector)))


def iter_by_strategies(state_list: list, p_list: list, r_list: list, max_steps=100):
    # выбираем произвольную стратегию
    new_opt_strat = 0
    prev_opt_strat = new_opt_strat

    for i in range(max_steps):
        print(f'--iteration step {i}--')
        Im = np.identity(len(state_list))
        pk_matrix = p_list[new_opt_strat]
        rk_matrix = r_list[new_opt_strat]
        # Этап оценивания параметров
        # 1.Вычисление ожидаемого дохода за один шаг при k-ой стационарной стратегии
        vi = [(pk_matrix[n_from, :] * rk_matrix[n_from, :]).sum() for n_from in range(len(state_list))]

        # 2. Решаем систему
        a = (Im - pk_matrix)

        # Fr(m)=0
        a = np.delete(a, len(state_list) - 1, 1)

        # добавляем Er
        a = np.c_[np.array([1] * len(state_list)), a]

        # находим значения Er, Fr1, Fr2
        solve_res = np.linalg.lstsq(a=a, b=vi, rcond=None)
        stnry_vector = solve_res[0]

        print_iter_by_strat(vi=vi, stnry_vector=stnry_vector)

        # TODO Этап улучшения стратегии
        new_opt_strat = np.random.randint(0, 4)

        if new_opt_strat == prev_opt_strat:
            print('\nOPTIMAL STRATEGY FOUND!')
            return new_opt_strat
        else:
            print('keep finding\n')
            prev_opt_strat = new_opt_strat

    return new_opt_strat


if __name__ == '__main__':
    # допустимые решения
    conditions = [
        {
            'G': 'X1',
            'desc': 'не применять',
            'P': np.array([
                [0.2, 0.5, 0.3],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0]]),
            'R': np.array([
                [7, 6, 3],
                [0, 5, 1],
                [0, 0, -1]])
        },
        {
            'G': 'X2',
            'desc': 'применять',
            'P': np.array([
                [0.3, 0.6, 0.1],
                [0.1, 0.6, 0.3],
                [0.05, 0.4, 0.55]]),
            'R': np.array([
                [6, 5, -1],
                [7, 4, 0],
                [6, 3, -2]])
        }
    ]

    # список состояний
    state_list = {
        "s1": "хорошее",
        "s2": "удовлетворительное",
        "s3": "плохое"
    }

    strat_len = len(conditions) ** len(state_list)
    p_list = []
    r_list = []
    desc_list = []
    for code in range(strat_len):
        bin_vect = [(code & (1 << shift)) >> shift for shift in range(len(state_list))]

        p_list.append(np.array([conditions[bin_vect[si]]['P'][si] for si in range(len(bin_vect))]))
        r_list.append(np.array([conditions[bin_vect[si]]['R'][si] for si in range(len(bin_vect))]))

        curr_desc = ''.join([f"{conditions[bin_vect[si]]['G']}" for si in range(len(bin_vect))])
        desc_list.append(curr_desc)

    best_strategy_i = brood_force(
        state_list=state_list,
        p_list=p_list,
        r_list=r_list
    )
    print(f"\n\nbrood_force: best_strategy = '{desc_list[best_strategy_i]}'\n".upper())

    best_strategy_i = iter_by_strategies(
        state_list=state_list,
        p_list=p_list,
        r_list=r_list
    )
    print(f"\n\niteration: best_strategy = '{desc_list[best_strategy_i]}'\n".upper())
