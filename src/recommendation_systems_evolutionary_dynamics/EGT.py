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

    def plot_transition_matrix2(self, matrix, threshold, scale=1):
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

    def plot_transition_matrix(self, matrix, states, scale=1):
        """
            matrix: 8x8 numpy array (transition probabilities)
            states: list of 8 state labels, e.g. ['DDD', 'DDC', ... 'CCC']
            """
        matrix = copy.deepcopy(matrix)
        G = nx.DiGraph()

        # Add nodes
        for s in states:
            G.add_node(s)

        # Break ties by marking the lesser transitions as -1
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] > matrix[j, i]:
                    matrix[j, i] = -1
                else:
                    matrix[i, j] = -1

        # Add edges where hamming distance == 1
        for i, s_i in enumerate(states):
            for j, s_j in enumerate(states):
                if i != j and matrix[i, j] > -1 and self._hamming_dist(states[i], states[j]) == 1:
                    G.add_edge(s_i, s_j, weight=matrix[i, j] * scale)

        if len(matrix) == len(matrix[0]) == len(states) == 8:
            coords = [(0.0, -0.2), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -0.2), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)]
        else:
            assert False, "Wrong states or matrix dimensions"
        pos = {states[i]: coords[i] for i in range(len(states))}

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

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color="lightgray",
            edgecolors="black",
        )

        # Draw edges, crucially passing node_size=node_sizes to offset arrows
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="->",
            arrowsize=25,
            node_size=node_sizes,  # <-- important to include
            min_source_margin=15,
            min_target_margin=15,
            width=2,
            edge_color='black',
            connectionstyle='arc3,rad=0'  # slight curve helps visibility
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Edge labels
        edge_labels = {
            (u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.axis("off")
        plt.show()

    @staticmethod
    def compute_stationary_distribution(matrix):
        # Compute eigenvalues and right eigenvectors of the transpose of P.
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)

        # Find the index of the eigenvalue closest to 1.
        idx = np.argmin(np.abs(eigenvalues - 1))

        # Extract the corresponding eigenvector and take the real part (if it is complex)
        stationary = eigenvectors[:, idx].real

        # Normalize the eigenvector so that the sum of its components equals 1.
        stationary = stationary / np.sum(stationary)

        return stationary

    @staticmethod
    def visualize_stationary_distribution_bar_charts(stationary_dist, states=None, actions_labels=None, actions_names=None, players_names=None, ylabel=None):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        ax_all = axes[0, 0]
        x_positions = np.arange(len(states))
        ax_all.bar(x_positions, stationary_dist, color='gray')
        ax_all.set_xticks(x_positions)
        ax_all.set_xticklabels(states, rotation=45)
        ax_all.set_ylim([0, 1])
        ax_all.set_title("All Populations")
        ax_all.set_ylabel(ylabel)

        def cooperator_defector_fraction(stationary_dist, states, actions_labels, player_index):
            strategies_fractionss = np.zeros(len(actions_labels))
            for i, st in enumerate(states):
                strategies_fractionss[actions_labels.index(st[player_index])] += stationary_dist[i]

            return strategies_fractionss

        # 2) For each of the three sectors, do a bar chart of (cooperators, defectors)
        #    We assume sector 0 = public, 1 = private, 2 = civil, matching your naming
        #    but you can adapt as needed.

        # sector 0 -> public
        ax_pub = axes[0, 1]
        strategies_fractionss = cooperator_defector_fraction(stationary_dist, states, actions_labels, player_index=0)
        ax_pub.bar([0, 1], strategies_fractionss, color=['blue', 'red'])
        ax_pub.set_xticks([0, 1])
        ax_pub.set_xticklabels(actions_names)
        ax_pub.set_ylim([0, 1])
        ax_pub.set_title(players_names[0].capitalize())

        # sector 1 -> private
        ax_priv = axes[1, 0]
        strategies_fractionss = cooperator_defector_fraction(stationary_dist, states, actions_labels, player_index=1)
        ax_priv.bar([0, 1], strategies_fractionss, color=['blue', 'red'])
        ax_priv.set_xticks([0, 1])
        ax_priv.set_xticklabels(actions_names)
        ax_priv.set_ylim([0, 1])
        ax_priv.set_title(players_names[1].capitalize())

        # sector 2 -> civil
        ax_civ = axes[1, 1]
        strategies_fractionss = cooperator_defector_fraction(stationary_dist, states, actions_labels, player_index=2)
        ax_civ.bar([0, 1], strategies_fractionss, color=['blue', 'red'])
        ax_civ.set_xticks([0, 1])
        ax_civ.set_xticklabels(actions_names)
        ax_civ.set_ylim([0, 1])
        ax_civ.set_title(players_names[2].capitalize())

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _fermi_rule(p1_fitness: float, p2_fitness: float, beta: float) -> float:
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
            
        elif process == "pairwise" and not sync:
            res = self._pairwise_async(reps, steps, beta, u, return_hist, print_rep_interval)
            
        elif process == "pairwise" and sync:
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
                    if random.random() < self._fermi_rule(fitness_i, fitness_j, beta):
                        if random.random() < u:
                            new_P[i] = (new_P[i] + random.randint(0, len(self._actions) - 1)) % len(
                                self._actions)
                        else:
                            new_P[i] = strat_j

                    new_strategies_fractions = self._compute_fractions_from_pop([new_P])
                    strategies_fractionss_hist[r, t + 1, s] = new_strategies_fractions[0]
                    Ps_hist[s][r, t + 1] = np.array(new_P)
                    prev_Ps[s] = new_P
                    prev_strategies_fractionss[s] = new_strategies_fractions[0]

        mean_fractionss_hist = np.mean(strategies_fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _death_birth_sync(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
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
                        j = random.randint(0, len(new_Ps[s]) - 1)
                        strat_i, strat_j = prev_Ps[s][i], prev_Ps[s][j]
                        fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                        fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                        if random.random() < self._fermi_rule(fitness_i, fitness_j, beta):
                            if random.random() < u:
                                new_Ps[s][i] = (new_Ps[s][i] + random.randint(0, len(self._actions) - 1)) % len(
                                    self._actions)
                            else:
                                new_Ps[s][i] = strat_j

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

    def _pairwise_async(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
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
                    else:
                        if random.random() < u:
                            new_P[i] = (new_P[i] + random.randint(0, len(self._actions) - 1)) % len(
                                self._actions)
                        else:
                            new_P[i] = strat_j

                    new_strategies_fractions = self._compute_fractions_from_pop([new_P])
                    strategies_fractionss_hist[r, t + 1, s] = new_strategies_fractions[0]
                    Ps_hist[s][r, t + 1] = np.array(new_P)
                    prev_Ps[s] = new_P
                    prev_strategies_fractionss[s] = new_strategies_fractions[0]

        mean_fractionss_hist = np.mean(strategies_fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _pairwise_sync(self, rep: int = 1, steps: int = 50, beta: float = 1, u: float = 0.01,
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
                        j = random.randint(0, len(new_Ps[s]) - 1)
                        strat_i, strat_j = prev_Ps[s][i], prev_Ps[s][j]
                        fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                        fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                        if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                            if random.random() < u:
                                new_Ps[s][i] = (new_Ps[s][i] + random.randint(0, len(self._actions) - 1)) % len(
                                    self._actions)
                            else:
                                new_Ps[s][i] = strat_j

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

    def visualize_evol(self, fractionss_hist, players: list[str], xlabel: str = None,
                       ylabel: str = None, title: str = None):
        timesteps = np.arange(fractionss_hist.shape[0])
        plt.figure(figsize=(10, 6))
        for col in range(fractionss_hist.shape[1]):
            plt.plot(timesteps, fractionss_hist[:, col], label=players[col])
        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_stationary_dist(self, P_hist,  xlabel: str = None,
                                  ylabel: str = None, title: str = None):
        states = P_hist.sum(axis=2).flatten().astype(int)
        unique_states, counts = np.unique(states, return_counts=True)
        fractions = counts / len(states)
        most_common_idx = int(np.argmax(fractions))
        plt.plot(unique_states, fractions)
        plt.scatter(unique_states[most_common_idx], fractions[most_common_idx], color="red",
                    label=f"Most Common State ({unique_states[most_common_idx]}, {fractions[most_common_idx]:.4f})")
        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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

    def gradient_of_selection(self, config, beta, mu):
        gradients = []
        n_pop = self._n
        fractions = []
        for i in range(n_pop):
            Z = self._Zs[i]
            j = config[i]
            # Avoid division by zero; we assume Z>=2
            fractions.append([j / Z, (Z - j) / Z])

        # For each population, compute the fitness for cooperators and defectors,
        # then compute the gradient of selection using the formula.
        for i in range(n_pop):
            # Strategy 0 is assumed to be Cooperator, strategy 1 is Defector.
            f_C = self.compute_fitness(i, 1, fractions)
            f_D = self.compute_fitness(i, 0, fractions)
            Z = self._Zs[i]
            j = config[i]
            # Term from imitation based on fitness differences.
            # We use j/(Z-1) provided Z > 1.
            if Z > 1:
                term1 = ((Z - j) / Z) * (j / (Z - 1))
            else:
                term1 = 0
            # Term due to mutation (or exploration).
            term2 = ((Z - 2 * j) / Z) * mu

            # The hyperbolic tangent ensures the term scales between -1 and 1.
            gradient_i = term1 * math.tanh(beta * (f_C - f_D) / 2) + term2
            gradients.append(gradient_i)

        return gradients

    def plot_gradient_field(self, beta, mu, title=None, players_names=None, fraction_name=None, legend=None):
        if self._n != 3:
            print("plot_gradient_field is implemented for 3 populations only.")
            return

            # Population sizes for the three populations.
        Z1, Z2, Z3 = self._Zs[0], self._Zs[1], self._Zs[2]

        # Lists to store fraction coordinates and gradient components.
        X, Y, Z_coords = [], [], []
        U, V, W = [], [], []

        # Iterate over all possible configurations: j1 from 0 to Z1, j2 from 0 to Z2, j3 from 0 to Z3.
        for j1 in range(Z1 + 1):
            for j2 in range(Z2 + 1):
                for j3 in range(Z3 + 1):
                    config = [Z1-j1, Z2-j2, Z3-j3]

                    grad = self.gradient_of_selection(config, beta, mu)
                    # Convert counts to fractions.
                    x = (Z1-j1) / Z1
                    y = (Z2-j2) / Z2
                    z = (Z3-j3) / Z3

                    X.append(x)
                    Y.append(y)
                    Z_coords.append(z)
                    U.append(grad[0])
                    V.append(grad[1])
                    W.append(grad[2])

        # Convert lists to NumPy arrays.
        X = np.array(X)
        Y = np.array(Y)
        Z_coords = np.array(Z_coords)
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)

        # Compute the magnitude of the gradient for each configuration.
        mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)
        # Normalize magnitudes for the colormap.
        norm = plt.Normalize(vmin=mag.min(), vmax=mag.max())
        # Use a colormap that goes from blue (low) to red (high). You can choose another cmap if desired.
        colors = plt.cm.jet(norm(mag))

        # Create a 3D quiver plot.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot arrows: each arrow is located at (X, Y, Z_coords) with components (U, V, W)
        q = ax.quiver(X, Y, Z_coords, U, V, W, length=0.05, normalize=True, color=colors)

        ax.set_xlabel(f"Fraction of {fraction_name} in {players_names[0]}")
        ax.set_ylabel(f"Fraction of {fraction_name} in {players_names[1]}")
        ax.set_zlabel(f"Fraction of {fraction_name} in {players_names[2]}")
        ax.set_title(title)

        # Create a ScalarMappable for the colorbar using the same colormap and normalization.
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])  # Dummy array for the ScalarMappable.
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label(legend)
        plt.show()

    def run_replicator_dynamics(self, steps=50,  players_names=None, actions_names=None):
        # Convert the current fractions to a NumPy array of shape (n_pops, n_strats)
        fractions = np.array(self._strategies_fractionss, dtype=float)

        # Store history of fractions: shape (time, pop, strategy)
        fractions_hist = np.zeros((steps + 1, self._n, len(self._actions)))
        fractions_hist[0] = fractions

        for t in range(steps):
            new_fractions = np.zeros_like(fractions)

            # For each population i
            for i in range(self._n):
                # 1) Compute payoff to each strategy k in population i
                payoffs = []
                for k in range(len(self._actions)):
                    payoff_k = self.compute_fitness(i, k, fractions)
                    payoffs.append(payoff_k)
                payoffs = np.array(payoffs, dtype=float)

                # 2) (Optional) SHIFT payoffs if some are negative or if avg payoff might be <= 0
                #    so that average payoff is safely positive:
                shift_amount = -payoffs.min()
                if shift_amount >= 0:
                    payoffs += (shift_amount + 1e-12)

                # 3) Compute the average payoff in population i:
                #    dot product of fractions[i] with the payoffs for each strategy
                avg_payoff_i = np.dot(fractions[i], payoffs)

                # 4) Replicator update:
                #    x_{i,k}(t+1) = x_{i,k}(t)*payoffs[k] / avg_payoff_i
                if avg_payoff_i > 1e-12:
                    new_fractions[i] = fractions[i] * payoffs / avg_payoff_i
                else:
                    # If the average payoff is ~0, just hold the old distribution
                    new_fractions[i] = fractions[i]

            # 5) Update fractions for the next step
            fractions = new_fractions
            fractions_hist[t + 1] = fractions

        # --- Plotting ---
        fig, axes = plt.subplots(1, self._n, figsize=(6 * self._n, 4), sharey=True)
        if self._n == 1:
            axes = [axes]  # Make it iterable if there's only one population

        time_points = range(steps + 1)
        for i in range(self._n):
            ax = axes[i]
            for k in range(len(self._actions)):
                ax.plot(time_points, fractions_hist[:, i, k], label=f"{actions_names[k]}")
            ax.set_title(f"{players_names[i]}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

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