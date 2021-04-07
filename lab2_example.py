import numpy as np

from lab_2_finit import get_finit_table_list
from lab_2_inf import inf_brood_force, inf_iter_by_strategies

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

    best_strategy_i = inf_brood_force(
        state_list=state_list,
        p_list=p_list,
        r_list=r_list
    )
    print(f"\n\nbrood_force: best_strategy = '{desc_list[best_strategy_i]}'\n".upper())

    best_strategy_i = inf_iter_by_strategies(
        state_list=state_list,
        p_list=p_list,
        r_list=r_list
    )
    print(f"\n\niteration: best_strategy = '{desc_list[best_strategy_i]}'\n".upper())

    table_list = get_finit_table_list(
        N=3,
        g_list=[cond['G'] for cond in conditions],
        state_list=list(state_list.keys()),
        p_list=[cond['P'] for cond in conditions],
        r_list=[cond['R'] for cond in conditions],
        a=1
    )
