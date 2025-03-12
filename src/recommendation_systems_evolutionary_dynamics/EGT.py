import math
import random
from itertools import product
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt


class Game:
    def __init__(self, Zs: list[int], strategies_counts: list[list[int]] = None, strategies_fractionss: list[list[float]] = None, payoff_matrix=None, players_names=None, actions_names=None):
        self._Zs = Zs  # Population sizes for each player
        self._n = len(Zs)  # Number of players

        self._payoff_matrix = None

        if ((strategies_counts is not None) and (strategies_fractionss is not None) or
                ((strategies_counts is None) and (strategies_fractionss is None))):
            assert False, "Choose either strategies fractions or counts"

        if strategies_fractionss is not None:
            self._strategies_fractionss = copy.deepcopy(strategies_fractionss)
            self._strategies_counts = self._count_strategies_from_fractions(strategies_fractionss)
        elif strategies_counts is not None:
            self._strategies_counts = copy.deepcopy(strategies_counts)
            self._strategies_fractionss = self._compute_fractions_from_counts(strategies_counts)

        # Derive actions from the first fraction list length.
        self._actions = list(range(len(self._strategies_fractionss[0])))

        # Initialize populations for each player.
        self._Ps = self._init_populations()

        if payoff_matrix is None:
            self._init_payoff_matrix()
        else:
            assert self._parse_payoff_matrix(payoff_matrix), "Invalid payoff matrix format."
            self._payoff_matrix = payoff_matrix

        if players_names is None:
            self._players_names = ["player_"+str(i) for i in range(1, len(Zs)+1)]
        else:
            assert self._parse_players_names(players_names), "Invalid players names"
            self._players_names = copy.deepcopy(players_names)

        if actions_names is None:
            self._actions_names = ["action_"+str(i) for i in range(1, len(self._strategies_fractionss[0])+1)]
        else:
            assert self._parse_actions_names(actions_names), "Invalid actions names"
            self._actions_names = copy.deepcopy(actions_names)

    def _parse_actions_names(self, actions_names):
        if not isinstance(actions_names, list):
            return False
        if len(actions_names) != len(self._strategies_fractionss[0]):
            return False
        if not all(isinstance(x, str) for x in actions_names):
            return False

        return True

    def _parse_players_names(self, players_names):
        if not isinstance(players_names, list):
            return False
        if len(players_names) != len(self._Zs):
            return False
        if not all(isinstance(x, str) for x in players_names):
            return False

        return True

    def _parse_payoff_matrix(self, payoff_matrix: dict) -> bool:
        if not isinstance(payoff_matrix, dict):
            print("Payoff matrix must be a dict")
            return False
        if len(payoff_matrix) != len(self._actions) ** self._n:
            print("Payoff matrix must must have (number of actions ^ number of player) entries")
            return False

        for key, value in payoff_matrix.items():
            if not (isinstance(key, tuple)):
                print("Key must be a tuple")
                return False
            if len(key) != self._n:
                print("Key size must = number of players")
                return False
            if not (all(isinstance(x, int) for x in key)):
                print("Key tuple must contain int")
                return False
            if not (isinstance(value, list)):
                print("Value must be a list")
                return False
            if not all(type(x) in (int, float) for x in value):
                print("Value list must contain int or float")
                return False
            if not (len(value) == self._n):
                print("Invalid value length")
                return False

        return True

    @staticmethod
    def _init_population(counts: list[int]) -> list[int]:
        """Initialize a single population of size N based on the provided fractions."""
        population = []
        for strategy, count in enumerate(counts):
            population.extend([strategy] * count)
        random.shuffle(population)
        return population

    def _init_populations(self):
        """Initialize populations for all players."""
        return [self._init_population(counts) for counts in self._strategies_counts]

    def _init_payoff_matrix(self):
        """Create a default payoff matrix with zero payoffs."""
        strat_combinations = list(product(self._actions, repeat=self._n))
        self._payoff_matrix = {tuple(combo): [0] * self._n for combo in strat_combinations}

    def compute_fitness(self, player, strategy, strategies_fractionss):
        all_strats = []
        for sec_idx in range(len(strategies_fractionss)):
            if sec_idx == player:
                # The player uses exactly one strategy (strategy)
                all_strats.append([strategy])
            else:
                # This player can use any of its strategies
                num_strats = len(strategies_fractionss[sec_idx])
                all_strats.append(range(num_strats))

        payoff_sum = 0.0
        for combo in itertools.product(*all_strats):
            prob = 1.0
            for sec_idx, s_idx in enumerate(combo):
                if sec_idx != player:
                    prob *= strategies_fractionss[sec_idx][s_idx]

            # Retrieve the payoff tuple from the payoff matrix
            payoffs = self._payoff_matrix[combo]  # e.g. (p0, p1, p2)

            # Add to the sum: Probability * payoff for the player
            payoff_sum += prob * payoffs[player]

        return payoff_sum

    def _fixation_prob(self, player, i, j, beta, strategies_fractions):
        p_ij = 0
        N = self._Zs[player]
        strategies_fractions = copy.deepcopy(strategies_fractions)

        for l in range(1, N):
            p = 1
            for k in range(1, l + 1):
                strategies_fractions[player][i] = 1 - k/N
                strategies_fractions[player][j] = k/N

                p1_fitness = self.compute_fitness(player, i, strategies_fractions)
                p2_fitness = self.compute_fitness(player, j, strategies_fractions)
                T_k_minus = (k / N) * ((N - k) / N) * self._fermi_rule(p2_fitness, p1_fitness, beta)
                T_k_plus = (k / N) * ((N - k) / N) * self._fermi_rule(p1_fitness, p2_fitness, beta)
                p *= T_k_minus / T_k_plus
            p_ij += p
        p_ij = 1 / (1 + p_ij)
        return p_ij

    @staticmethod
    def _hamming_dist(a, b):
        dist = 0
        for i in range(len(a)):
            if a[i] != b[i]:
                dist += 1
        return dist

    @staticmethod
    def _diff_ind(a, b):
        diff = []
        for i in range(len(a)):
            if a[i] != b[i]:
                diff.append(i)
        return diff

    def compute_trans_matrix(self, beta):
        states = list(product(self._actions, repeat=self._n))
        num_states = len(states)
        m = np.zeros((num_states, num_states))
        for i in range(num_states):
            state_i = states[i]
            for j in range(num_states):
                state_j = states[j]
                if i == j:
                    continue

                Ps_fracs = [[0] * len(self._actions) for _ in range(len(self._Zs))]
                for player, strat in enumerate(state_i):
                    Ps_fracs[player][strat] = 1

                player = self._diff_ind(state_i, state_j)[0]
                fixation_prob = self._fixation_prob(player, state_i[player], state_j[player], beta,
                                                    Ps_fracs)

                m[i, j] = fixation_prob / self._n
        for i in range(len(m)):
            s = 0
            for j in range(len(m)):
                if i != j:
                    s += m[i, j]
            m[i, i] = 1 - s
        return m

    def plot_transition_matrix(self, matrix, threshold, scale=1):
        """
        matrix: 8x8 numpy array (transition probabilities)
        states: list of 8 state labels, e.g. ['DDD', 'DDC', ... 'CCC']
        """
        matrix = copy.deepcopy(matrix)
        G = nx.DiGraph()
        states = ["DDD", "DDC", "DCD", "DCC", "CDD", "CDC", "CCD", "CCC"]

        # Add nodes
        for s in states:
            G.add_node(s)

        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] > matrix[j, i]:
                    matrix[j, i] = -1
                else:
                    matrix[i, j] = -1

        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states):
                if i != j and matrix[i, j] > -1 and self._hamming_dist(states[i], states[j]) == 1:
                    G.add_edge(s_i, s_j, weight=matrix[i, j]*scale)

        pos = {
            "DDD": (0.0, 0.0),
            "DDC": (1.0, 0.0),
            "DCD": (0.0, 1.0),
            "DCC": (1.0, 1.0),
            "CDD": (2.0, 0.0),
            "CDC": (2.0, 1.0),
            "CCD": (1.0, 2.0),
            "CCC": (2.0, 2.0),
        }
        # 2) Compute incoming weight sums
        in_weight_sum = {}
        for node in G.nodes():
            total_in = 0.0
            for pred in G.predecessors(node):
                edge_data = G.get_edge_data(pred, node)
                if edge_data is not None and "weight" in edge_data:
                    total_in += edge_data["weight"]
            in_weight_sum[node] = total_in

        # 3) Scale node sizes
        min_size = 300
        scale_factor = 3000
        node_sizes = [
            min_size + in_weight_sum[node] * scale_factor
            for node in G.nodes()
        ]

        # 4) Plot

        plt.figure(figsize=(8, 6))

        # Suppose you computed node_sizes in some way
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color="lightgray",
            edgecolors="black",
        )

        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="->",
            arrowsize=25,
            min_source_margin=15,
            min_target_margin=15,
            width=2,
            edge_color='black',
        )

        nx.draw_networkx_labels(G, pos, font_size=10)

        edge_labels = {
            (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.axis("off")
        plt.show()

    @staticmethod
    def _fermi_rule(p1_fitness: float, p2_fitness: float, beta: float) -> float:
        """Return the probability of imitation using the Fermi rule."""
        return 1 / (1 + math.exp(-beta * (p2_fitness - p1_fitness)))

    def moran_process(self, process, sync=False, steps: int = 50, reps: int = 1, beta: float = 1, u: float = 0.01, return_hist: bool = False, print_rep_interval: int = None):
        if process == "bd" and sync:
            res = self._birth_death_sync(reps, steps, beta, u, return_hist, print_rep_interval)

        elif process == "bd" and not sync:
            res = self._birth_death_async(reps, steps, beta, u, return_hist, print_rep_interval)

        elif process == "db" and sync:
            res = self._death_birth_sync(reps, steps, beta, u, return_hist, print_rep_interval)

        elif process == "db" and not sync:
            res = self._death_birth_async(reps, steps, beta, u, return_hist, print_rep_interval)
            
        elif process == "pairwise" and sync:
            res = self._pairwise_sync(reps, steps, beta, u, return_hist, print_rep_interval)
            
        elif process == "pairwise" and not sync:
            res = self._pairwise_sync(reps, steps, beta, u, return_hist, print_rep_interval)
        else:
            assert False, "Wrong process name"
        return res

    def _birth_death_async(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
                           return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = np.zeros((rep, steps + 1, self._n, len(self._actions)))
        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[r, 0, s] = self._strategies_fractionss[s]

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_strategies_fractionss = [fracs.copy() for fracs in self._strategies_fractionss]

            for t in range(steps):
                for s in range(self._n):
                    new_P = copy.deepcopy(prev_Ps[s])

                    i, j = random.sample(range(len(new_P)), 2)
                    strat_i, strat_j = new_P[i], new_P[j]
                    fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                    if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                        if random.random() < u:
                            new_P[j] = (new_P[j] + random.randint(0, len(self._actions) - 1)) % len(
                                self._actions)
                        else:
                            new_P[j] = strat_i

                    new_strategies_fractions = self._compute_fractions_from_pop([new_P])
                    strategies_fractionss_hist[r, t + 1, s] = new_strategies_fractions[0]
                    Ps_hist[s][r, t + 1] = np.array(new_P)
                    prev_Ps[s] = new_P
                    prev_strategies_fractionss[s] = new_strategies_fractions[0]

        mean_fractionss_hist = np.mean(strategies_fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _birth_death_sync(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
                           return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = np.zeros((rep, steps + 1, self._n, len(self._actions)))
        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[r, 0, s] = self._strategies_fractionss[s]

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_strategies_fractionss = [fracs.copy() for fracs in self._strategies_fractionss]

            for t in range(steps):
                new_Ps = copy.deepcopy(prev_Ps)
                for s in range(self._n):
                    for i in range(len(new_Ps[s])):
                        j = random.randint(0, len(new_Ps[s])-1)
                        strat_i, strat_j = prev_Ps[s][i], prev_Ps[s][j]
                        fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                        fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                        if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                            if random.random() < u:
                                new_Ps[s][j] = (new_Ps[s][j] + random.randint(0, len(self._actions) - 1)) % len(
                                    self._actions)
                            else:
                                new_Ps[s][j] = strat_i

                prev_Ps = new_Ps
                new_strategies_fractions = self._compute_fractions_from_pop(new_Ps)
                prev_strategies_fractionss = new_strategies_fractions
                strategies_fractionss_hist[r, t + 1] = new_strategies_fractions
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(new_Ps[s])

        mean_fractionss_hist = np.mean(strategies_fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _death_birth_async(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
                    return_hist: bool = False, print_rep_interval: int = None):
        """
        Run the birth–death (imitation–mutation) process simulation.

        Args:
            rep: Number of replicates.
            steps: Number of time steps per replicate.
            beta: Selection intensity for the Fermi rule.
            u: Mutation probability.
            return_hist: Whether to return the full history.
            print_rep_interval: Interval at which to print replicate progress.

        Returns:
            The mean fraction history over replicates and, optionally, the full history.
        """
        # History arrays: fractionss_hist shape: (rep, steps+1, players, 2 strategies)
        fractionss_hist = np.zeros((rep, steps + 1, self._n, 2))
        # Ps_hist for each player: shape: (rep, steps+1, population size)
        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            if print_rep_interval is not None and r % print_rep_interval == 0:
                print(f"Replicate: {r}")
            # Use shallow copies since populations and fractions are lists of primitives.
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_fractionss = [fracs.copy() for fracs in self._strategies_fractionss]

            # Record initial state.
            for s in range(self._n):
                Ps_hist[s][r, 0] = np.array(prev_Ps[s])
                fractionss_hist[r, 0, s] = np.array(prev_fractionss[s])

            for t in range(steps):
                # Process each player.
                for s in range(self._n):
                    population = prev_Ps[s]
                    pop_size = len(population)

                    # Imitation: select two distinct individuals.
                    i, j = random.sample(range(pop_size), 2)
                    strat_i, strat_j = population[i], population[j]
                    # Compute fitness using current fractions (x, y, z correspond to players 0, 1, 2 respectively).
                    fitness_i = self.compute_fitness(s, strat_i, prev_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_fractionss)
                    if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                        # Mutation: select one individual randomly and flip its strategy with probability u.
                        if random.random() < u:
                            population[j] = (population[j] + random.randint(0, len(self._actions)-1)) % len(self._actions)
                        else:
                            population[j] = strat_i

                    # Update fraction for the current player (assuming binary strategies: 0 and 1).
                    fraction_strategy1 = np.sum(np.array(population) == 1) / self._Zs[s]
                    prev_fractionss[s][1] = fraction_strategy1
                    prev_fractionss[s][0] = 1 - fraction_strategy1

                # Record state after processing all players.
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(prev_Ps[s])
                fractionss_hist[r, t + 1] = np.array(prev_fractionss)

        mean_fractionss_hist = np.mean(fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _death_birth_sync(self, rep: int = 1, steps: int = 1, beta: float = 1, u: float = 0.01,
                    return_hist: bool = False, print_rep_interval: int = None):
        """
        Run the birth–death (imitation–mutation) process simulation.

        Args:
            rep: Number of replicates.
            steps: Number of time steps per replicate.
            beta: Selection intensity for the Fermi rule.
            u: Mutation probability.
            return_hist: Whether to return the full history.
            print_rep_interval: Interval at which to print replicate progress.

        Returns:
            The mean fraction history over replicates and, optionally, the full history.
        """
        # History arrays: fractionss_hist shape: (rep, steps+1, players, 2 strategies)
        fractionss_hist = np.zeros((rep, steps + 1, self._n, 2))
        # Ps_hist for each player: shape: (rep, steps+1, population size)
        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            if print_rep_interval is not None and r % print_rep_interval == 0:
                print(f"Replicate: {r}")
            # Use shallow copies since populations and fractions are lists of primitives.
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_fractionss = [fracs.copy() for fracs in self._strategies_fractionss]

            # Record initial state.
            for s in range(self._n):
                Ps_hist[s][r, 0] = np.array(prev_Ps[s])
                fractionss_hist[r, 0, s] = np.array(prev_fractionss[s])

            for t in range(steps):
                # Process each player.
                for s in range(self._n):
                    population = prev_Ps[s]
                    pop_size = len(population)

                    # Imitation: select two distinct individuals.
                    i, j = random.sample(range(pop_size), 2)
                    strat_i, strat_j = population[i], population[j]
                    # Compute fitness using current fractions (x, y, z correspond to players 0, 1, 2 respectively).
                    fitness_i = self.compute_fitness(s, strat_i, prev_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_fractionss)
                    if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                        # Mutation: select one individual randomly and flip its strategy with probability u.
                        if random.random() < u:
                            population[j] = (population[j] + random.randint(0, len(self._actions)-1)) % len(self._actions)
                        else:
                            population[j] = strat_i

                    # Update fraction for the current player (assuming binary strategies: 0 and 1).
                    fraction_strategy1 = np.sum(np.array(population) == 1) / self._Zs[s]
                    prev_fractionss[s][1] = fraction_strategy1
                    prev_fractionss[s][0] = 1 - fraction_strategy1

                # Record state after processing all players.
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(prev_Ps[s])
                fractionss_hist[r, t + 1] = np.array(prev_fractionss)

        mean_fractionss_hist = np.mean(fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _pairwise_async(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
                    return_hist: bool = False, print_rep_interval: int = None):
        """
        Run the birth–death (imitation–mutation) process simulation.

        Args:
            rep: Number of replicates.
            steps: Number of time steps per replicate.
            beta: Selection intensity for the Fermi rule.
            u: Mutation probability.
            return_hist: Whether to return the full history.
            print_rep_interval: Interval at which to print replicate progress.

        Returns:
            The mean fraction history over replicates and, optionally, the full history.
        """
        # History arrays: fractionss_hist shape: (rep, steps+1, players, 2 strategies)
        fractionss_hist = np.zeros((rep, steps + 1, self._n, 2))
        # Ps_hist for each player: shape: (rep, steps+1, population size)
        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            if print_rep_interval is not None and r % print_rep_interval == 0:
                print(f"Replicate: {r}")
            # Use shallow copies since populations and fractions are lists of primitives.
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_fractionss = [fracs.copy() for fracs in self._strategies_fractionss]

            # Record initial state.
            for s in range(self._n):
                Ps_hist[s][r, 0] = np.array(prev_Ps[s])
                fractionss_hist[r, 0, s] = np.array(prev_fractionss[s])

            for t in range(steps):
                # Process each player.
                for s in range(self._n):
                    population = prev_Ps[s]
                    pop_size = len(population)

                    # Imitation: select two distinct individuals.
                    i, j = random.sample(range(pop_size), 2)
                    strat_i, strat_j = population[i], population[j]
                    # Compute fitness using current fractions (x, y, z correspond to players 0, 1, 2 respectively).
                    fitness_i = self.compute_fitness(s, strat_i, prev_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_fractionss)
                    if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                        # Mutation: select one individual randomly and flip its strategy with probability u.
                        if random.random() < u:
                            population[j] = (population[j] + random.randint(0, len(self._actions)-1)) % len(self._actions)
                        else:
                            population[j] = strat_i

                    # Update fraction for the current player (assuming binary strategies: 0 and 1).
                    fraction_strategy1 = np.sum(np.array(population) == 1) / self._Zs[s]
                    prev_fractionss[s][1] = fraction_strategy1
                    prev_fractionss[s][0] = 1 - fraction_strategy1

                # Record state after processing all players.
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(prev_Ps[s])
                fractionss_hist[r, t + 1] = np.array(prev_fractionss)

        mean_fractionss_hist = np.mean(fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _pairwise_sync(self, rep: int = 1, steps: int = 1, beta: float = 1, u: float = 0.01,
                    return_hist: bool = False, print_rep_interval: int = None):
        """
        Run the birth–death (imitation–mutation) process simulation.

        Args:
            rep: Number of replicates.
            steps: Number of time steps per replicate.
            beta: Selection intensity for the Fermi rule.
            u: Mutation probability.
            return_hist: Whether to return the full history.
            print_rep_interval: Interval at which to print replicate progress.

        Returns:
            The mean fraction history over replicates and, optionally, the full history.
        """
        # History arrays: fractionss_hist shape: (rep, steps+1, players, 2 strategies)
        fractionss_hist = np.zeros((rep, steps + 1, self._n, 2))
        # Ps_hist for each player: shape: (rep, steps+1, population size)
        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            if print_rep_interval is not None and r % print_rep_interval == 0:
                print(f"Replicate: {r}")
            # Use shallow copies since populations and fractions are lists of primitives.
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_fractionss = [fracs.copy() for fracs in self._strategies_fractionss]

            # Record initial state.
            for s in range(self._n):
                Ps_hist[s][r, 0] = np.array(prev_Ps[s])
                fractionss_hist[r, 0, s] = np.array(prev_fractionss[s])

            for t in range(steps):
                # Process each player.
                for s in range(self._n):
                    population = prev_Ps[s]
                    pop_size = len(population)

                    # Imitation: select two distinct individuals.
                    i, j = random.sample(range(pop_size), 2)
                    strat_i, strat_j = population[i], population[j]
                    # Compute fitness using current fractions (x, y, z correspond to players 0, 1, 2 respectively).
                    fitness_i = self.compute_fitness(s, strat_i, prev_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_fractionss)
                    if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                        # Mutation: select one individual randomly and flip its strategy with probability u.
                        if random.random() < u:
                            population[j] = (population[j] + random.randint(0, len(self._actions)-1)) % len(self._actions)
                        else:
                            population[j] = strat_i

                    # Update fraction for the current player (assuming binary strategies: 0 and 1).
                    fraction_strategy1 = np.sum(np.array(population) == 1) / self._Zs[s]
                    prev_fractionss[s][1] = fraction_strategy1
                    prev_fractionss[s][0] = 1 - fraction_strategy1

                # Record state after processing all players.
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(prev_Ps[s])
                fractionss_hist[r, t + 1] = np.array(prev_fractionss)

        mean_fractionss_hist = np.mean(fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def visualize_evol(self, fractionss_hist, players: list[str], xlabel: str = None,
                       ylabel: str = None, title: str = None):
        timesteps = np.arange(fractionss_hist.shape[0])
        plt.figure(figsize=(10, 6))
        for col in range(fractionss_hist.shape[1]):
            plt.plot(timesteps, fractionss_hist[:, col], label=players[col])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_stationary_dist(self, Ps_hist, title: str = None):
        states = Ps_hist.sum(axis=2).flatten().astype(int)
        unique_states, counts = np.unique(states, return_counts=True)
        fractions = counts / len(states)
        most_common_idx = int(np.argmax(fractions))
        plt.plot(unique_states, fractions)
        plt.scatter(unique_states[most_common_idx], fractions[most_common_idx], color="red",
                    label=f"Most Common State ({unique_states[most_common_idx]}, {fractions[most_common_idx]:.4f})")
        plt.xlabel("State")
        plt.ylabel("Fraction")
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()

    def set_payoff_matrix(self, payoff_matrix: dict):
        assert self._parse_payoff_matrix(payoff_matrix), "Invalid Payoff Matrix"
        self._payoff_matrix = payoff_matrix

    def get_payoff_matrix(self) -> dict:
        return self._payoff_matrix

    def get_populations(self) -> list:
        return self._Ps

    def _count_strategies_from_fractions(self, strategies_fractionss):
        strategies_counts = []
        for i, pop in enumerate(strategies_fractionss):
            strategies_counts.append([])
            for frac in pop:
                strategies_counts[i].append(int(self._Zs[i] * frac))
            while sum(strategies_counts[i]) < self._Zs[i]:
                m = len(strategies_counts[i]) - 1
                strategies_counts[i][random.randint(0, m)] += 1
            while sum(strategies_counts[i]) > self._Zs[i]:
                m = len(strategies_counts[i]) - 1
                strategies_counts[i][random.randint(0, m)] -= 1
        return strategies_counts

    def _count_strategies_from_pop(self, Ps):
        strategies_counts = []

        for P in Ps:
            counts = [0]*(len(self._actions))
            for strat in P:
                counts[strat] += 1
            strategies_counts.append(counts)

        return strategies_counts

    def _compute_fractions_from_counts(self, strategies_counts):
        strategies_fractionss = []
        for i, pop in enumerate(strategies_counts):
            strategies_fractionss.append([])
            for count in pop:
                strategies_fractionss[i].append(round(float(count / self._Zs[i]), 4))
            if sum(strategies_fractionss[i]) != 1:
                m = random.randint(0, len(strategies_counts[i]) - 1)
                strategies_fractionss[i][m] = round(strategies_fractionss[i][m] + 1 - sum(strategies_fractionss[i]), 4)

        return strategies_fractionss

    def _compute_fractions_from_pop(self, Ps):
        strategies_count = self._count_strategies_from_pop(Ps)
        strategies_fractionss = self._compute_fractions_from_counts(strategies_count)

        return strategies_fractionss
    # Setters

    def set_strategies_counts(self, new_strategies_counts):
        self._strategies_counts = copy.deepcopy(new_strategies_counts)
        self._strategies_fractionss = self._compute_fractions_from_counts(self._strategies_counts)

    def set_strategies_fractionss(self, new_strategies_fractionss):
        self._strategies_fractionss = copy.deepcopy(new_strategies_fractionss)
        self._strategies_counts = self._count_strategies_from_fractions(self._strategies_counts)

    def set_players_names(self, new_players_names):
        assert self._parse_players_names(new_players_names), "Invalid players names"
        self._players_names = copy.deepcopy(new_players_names)

    def set_actions_names(self, new_actions_names):
        assert self._parse_actions_names(new_actions_names), "Invalid actions names"
        self._actions_names = copy.deepcopy(new_actions_names)


if __name__ == "__main__":
    pass