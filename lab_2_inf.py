import numpy as np


def print_brood_force(vi, stnry_vector):
    for i, curr_v in enumerate(vi):
        print(f"v{i + 1}k = {round(curr_v, 2)}")

    print()
    for i, curr_strny in enumerate(stnry_vector):
        print(f"П{i + 1}k = {round(curr_strny, 2)}")


def inf_brood_force(state_list: list, p_list: list, r_list: list):
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


def inf_iter_by_strategies(state_list: list, p_list: list, r_list: list, max_steps=100):
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
