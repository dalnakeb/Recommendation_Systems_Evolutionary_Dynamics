import numpy as np
import random
from itertools import product


def init_population(N: int, fractions: list[float]) -> np.array:
    strategies = [i for i in range(len(fractions))]
    Z = np.zeros(N, dtype=int)
    current_ind = 0
    for fraction_ind in range(0, len(fractions)):
        Z[current_ind:int(current_ind + int(N*fractions[fraction_ind]))] = strategies[fraction_ind]
        current_ind += int(N*fractions[fraction_ind])
    np.random.shuffle(Z)
    return Z


def init_populations(Ns: list[int], fractionss: list[list[float]]) -> np.array:
    Zs = []
    for N, fractions in zip(Ns, fractionss):
        Zs.append(init_population(N, fractions))

    return Zs


def init_payoff_matrix(strategies: list[int], n: int) -> dict:
    strat_combinations = list(product(strategies, repeat=n))
    payoff_matrix = {}
    for strat_combination in strat_combinations:
        payoff_matrix[strat_combination] = [0, 0, 0]

    return payoff_matrix


def compute_fitness(x, y, z, X, P, sector):
    f = 0
    if str.lower(sector) == "public":
        f = (y*z*P[[X, 1, 1][0]] +
             (1-y)*z*P[[X, 1, 0][0]] +
             y*(1-z)*P[[X, 0, 1][0]] +
             (1-y)*(1-z)*P[[X, 0, 0][0]])
    elif str.lower(sector) == "private":
        f = (x*z*P[[1, X, 1][1]] +
             (1-x)*z*P[[1, X, 0][1]] +
             x*(1-z)*P[[0, X, 1][1]] +
             (1-x)*(1-z)*P[[0, X, 0][1]])
    elif str.lower(sector) == "civil":
        f = (x*y*P[[1, 1, X][2]] +
             (1-x)*y*P[[1, 0, X][2]] +
             x*(1-y)*P[[0, 1, X][2]] +
             (1-x)*(1-y)*P[[0, 0, X][2]])

    return f


if __name__ == "__main__":
    print(init_payoff_matrix([0, 1], n=3))
