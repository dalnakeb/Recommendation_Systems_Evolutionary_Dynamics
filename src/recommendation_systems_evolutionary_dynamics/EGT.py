import math
import random
import networkx as nx
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt


class Game:
    def __init__(self, strategies_countss: list[list[int]] = None, payoff_matrix=None, players_names=None, actions_names=None):
        """
        :param strategies_countss: [[s11, s12], [s21, s22, s23], ...] where sij is the count of strategy j in population i
        :param payoff_matrix: {(s1i, s2j, ...): [p1i, p2j, ...]} map each strategy combination to list of payoffs
        :param players_names: [n1, n2, ...]
        :param actions_names: [a1, a2, ...]
        """
        self._Zs = [sum(strat_count) for strat_count in strategies_countss]  # Population sizes for each player
        self._n = len(self._Zs)  # Number of players

        self._strategies_countss = copy.deepcopy(strategies_countss)
        self._strategies_fractionss = self._compute_strategies_fractionss_from_countss(strategies_countss)

        assert self._parse_payoff_matrix(payoff_matrix), "Invalid payoff matrix format."
        self._payoff_matrix = payoff_matrix
        self._actions = [[i for i in range(len(strategies_fractions))] for strategies_fractions in self._strategies_fractionss]
        self._Ps = self._init_populations()

        if players_names is None:
            self._players_names = ["Player_"+str(i) for i in range(1, len(self._Zs)+1)]
        else:
            assert self._parse_players_names(players_names), "Invalid players names"
            self._players_names = copy.deepcopy(players_names)

        if actions_names is None:
            self._actions_names = ["Action_" + str(i+1) + str(j+1) for i in range(len(self._strategies_fractionss)) for j in range(len(self._strategies_fractionss[i]))]
        else:
            assert self._parse_actions_names(actions_names), "Invalid actions names"
            self._actions_names = copy.deepcopy(actions_names)

    def _parse_actions_names(self, actions_names):
        if not isinstance(actions_names, list):
            return False
        if not all(len(actions_names[i]) == len(self._strategies_fractionss[i]) for i in range(len(self._strategies_fractionss))):
            return False
        if not all(isinstance(x, str) for sublist in actions_names for x in sublist):
            return False

        return True

    def _parse_players_names(self, players_names):
        if not isinstance(players_names, list):
            return False
        if len(players_names) != len(self._strategies_fractionss):
            return False
        if not all(isinstance(x, str) for x in players_names):
            return False

        return True

    def _parse_payoff_matrix(self, payoff_matrix: dict) -> bool:
        if not isinstance(payoff_matrix, dict):
            print("Payoff matrix must be a dict")
            return False

        strat_comb = 1
        for i in range(len(self._strategies_fractionss)):
            strat_comb *= len(self._strategies_fractionss[i])

        if len(payoff_matrix) != strat_comb:
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
        return [self._init_population(counts) for counts in self._strategies_countss]

    def compute_fitness(self, player, strategy, strategies_fractionss):
        all_strats = []
        for sec_idx in range(len(strategies_fractionss)):
            if sec_idx == player:
                all_strats.append([strategy])
            else:
                num_strats = len(strategies_fractionss[sec_idx])
                all_strats.append(range(num_strats))

        payoff_sum = 0.0
        for combo in itertools.product(*all_strats):
            prob = 1.0
            for sec_idx, s_idx in enumerate(combo):
                if sec_idx != player:
                    prob *= strategies_fractionss[sec_idx][s_idx]

            payoffs = self._payoff_matrix[combo]

            payoff_sum += prob * payoffs[player]  # Expected Fitness

        return payoff_sum

    def _fixation_prob(self, player, i, j, beta, strategies_fractions):
        p_ij = 0
        N = self._Zs[player]

        # Precompute and cache fitness for each k = 1, ..., N-1.
        cache_i = {}
        cache_j = {}
        for k in range(1, N):
            copy_fractions = copy.deepcopy(strategies_fractions)
            copy_fractions[player][i] = 1 - k / N
            copy_fractions[player][j] = k / N
            cache_i[k] = self.compute_fitness(player, i, copy_fractions)
            cache_j[k] = self.compute_fitness(player, j, copy_fractions)

        for l in range(1, N):
            p = 1
            for k in range(1, l + 1):
                ratio = (self._fermi_rule(cache_j[k], cache_i[k], beta) /
                         self._fermi_rule(cache_i[k], cache_j[k], beta))
                p *= ratio
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
        states = list(self._payoff_matrix.keys())
        num_states = len(states)
        m = np.zeros((num_states, num_states))
        for i in range(num_states):
            state_i = states[i]
            for j in range(num_states):
                state_j = states[j]
                if i == j:
                    continue

                Ps_fracs = [[0] * len(self._actions[_]) for _ in range(len(self._Zs))]
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
        return m, states

    def plot_transition_matrix(self, matrix, states, actions_symbols, scale=1, save_file_name=None):
        matrix = copy.deepcopy(matrix)
        G = nx.DiGraph()
        states_string = []
        for state in states:
            state_string = ""
            for i in range(len(state)):
                state_string += actions_symbols[i][state[i]]
            states_string.append(state_string)

        # Add nodes
        for s in states_string:
            G.add_node(s)

        # Break ties by marking the lesser transitions as -1
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] > matrix[j, i]:
                    matrix[j, i] = -1
                elif matrix[i, j] < matrix[j, i]:
                    matrix[i, j] = -1

        # Add edges where hamming distance == 1
        dashed_edges = []
        arrow_edges = []
        for i, s_i in enumerate(states_string):
            for j, s_j in enumerate(states_string):
                if i != j and matrix[i, j] > -1 and self._hamming_dist(states_string[i], states_string[j]) == 1:
                    if matrix[i, j] != matrix[j, i]:
                        G.add_edge(s_i, s_j, weight=matrix[i, j] * scale)
                        arrow_edges.append((s_i, s_j))
                    else:
                        dashed_edges.append((s_i, s_j))
                        G.add_edge(s_i, s_j)

        # 2) Compute incoming weight sums
        in_weight_sum = {}
        for node in G.nodes():
            total_in = 0.0
            for pred in G.predecessors(node):
                edge_data = G.get_edge_data(pred, node)
                if edge_data is not None and "weight" in edge_data:
                    total_in += edge_data["weight"] / 4
            in_weight_sum[node] = total_in

        # 3) Scale node sizes
        min_size = 300
        scale_factor = 3000
        node_sizes = [
            min_size + in_weight_sum[node] * scale_factor
            for node in G.nodes()
        ]

        # 4) Plot
        plt.figure(figsize=(int(len(states_string)*1.5), int(len(states_string)*1.5*.8)))
        pos = nx.circular_layout(G)

        # Draw nodes
        # We can find the min and max to ensure the full colormap range is used
        node_colors = node_sizes / max(node_sizes)
        vmin = 0
        vmax = 1

        nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="black",
            cmap=plt.cm.Blues,  # A colormap (e.g. Blues, OrRd, etc.)
            vmin=vmin, vmax=vmax
        )

        arrow_len = 25 + 3 * (len(states_string) - 1)

        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="->",
            arrowsize=arrow_len,
            edgelist=arrow_edges,
            node_size=node_sizes,  # <-- important to include
            min_source_margin=15,
            min_target_margin=15,
            width=2,
            edge_color='black',
            connectionstyle='arc3,rad=0'  # slight curve helps visibility
        )

        nx.draw_networkx_edges(
            G,
            pos,
            style="dashed",
            edgelist=dashed_edges,
            width=2,
            edge_color='black',
            connectionstyle='arc3,rad=0'  # slight curve helps visibility
        )

        edge_labels = {}
        for u, v, d in G.edges(data=True):
            if "weight" in d:
                edge_labels[(u, v)] = f"{d['weight']:.2f}"
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_color='red',
            label_pos=0.4,
            font_size=22
        )

        for node, (x, y), c in zip(G.nodes(), pos.values(), node_colors):
            font_color = "white" if c > 0.5 else "black"
            plt.text(
                x, y,
                s=node,
                fontsize=24,  # change font size here
                color=font_color,  # contrast font color
                horizontalalignment='center',
                verticalalignment='center',
                zorder=10
            )

        plt.axis("off")
        if save_file_name is not None:
            plt.savefig(save_file_name)

        plt.show()

    def plot_transition_matrix_most_probable_route_from_i_to_j(
            self,
            matrix,
            states,
            actions_symbols,
            start_state,  # e.g. (0,0,0)
            end_state,  # e.g. (1,1,1)
            scale=1
    ):
        """
        Plots ONLY the single most-probable path (max product of transition probabilities)
        from `start_state` to `end_state`, showing only the nodes involved in that route.

        The cost for each edge is computed as:
            cost = -log(original_prob)
        where original_prob is the unscaled transition probability.
        This ensures non-negative costs for use in Dijkstra's algorithm.
        The scaled probability (original_prob * scale) is used for visualization.
        """
        matrix = copy.deepcopy(matrix)
        G = nx.DiGraph()

        # Convert each state into a string label
        states_string = []
        for state in states:
            label = "".join(actions_symbols[i][s] for i, s in enumerate(state))
            states_string.append(label)

        # Add all states as nodes
        for s in states_string:
            G.add_node(s)

        # Tie-breaking: for each pair, keep only the dominant transition.
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] > matrix[j, i]:
                    matrix[j, i] = -1
                elif matrix[i, j] < matrix[j, i]:
                    matrix[i, j] = -1

        # Build edges in G:
        #   - Only add edge if matrix[i, j] > -1 (i.e. it survived tie-breaking)
        #   - Only add if Hamming distance between labels is 1
        for i, s_i in enumerate(states_string):
            for j, s_j in enumerate(states_string):
                if i != j and matrix[i, j] > -1 and self._hamming_dist(s_i, s_j) == 1:
                    original_prob = matrix[i, j]
                    # Compute cost based on the original probability.
                    if original_prob <= 0:
                        cost = float('inf')
                    else:
                        cost = -math.log(original_prob)
                    # For visualization, use the scaled probability.
                    weight = original_prob * scale
                    G.add_edge(s_i, s_j, weight=weight, cost=cost)

        # Convert provided start and end states to labels
        start_label = "".join(actions_symbols[i][s] for i, s in enumerate(start_state))
        end_label = "".join(actions_symbols[i][s] for i, s in enumerate(end_state))

        if start_label not in G.nodes or end_label not in G.nodes:
            print(f"Start or end state not recognized: {start_label}, {end_label}")
            return

        # Find the most probable path using cost (i.e. maximizing the product of probabilities)
        try:
            path = nx.shortest_path(G, source=start_label, target=end_label, weight='cost')
        except nx.NetworkXNoPath:
            print(f"No path exists from {start_label} to {end_label}")
            return

        # Create a subgraph H with only the nodes and edges in the found path.
        path_edges = list(zip(path, path[1:]))
        H = nx.DiGraph()
        for node in path:
            H.add_node(node)
        for (u, v) in path_edges:
            if G.has_edge(u, v):
                H.add_edge(u, v, **G[u][v])

        # Compute node sizes based on incoming weights (for visualization)
        in_weight_sum = {}
        for node in H.nodes():
            total_in = 0.0
            for pred in H.predecessors(node):
                edge_data = H.get_edge_data(pred, node)
                if edge_data and "weight" in edge_data:
                    total_in += edge_data["weight"]
            in_weight_sum[node] = total_in

        min_size = 300
        scale_factor = 3000
        node_sizes = [1500 for node in H.nodes()]
        avg_node_size = sum(node_sizes) / len(node_sizes) if node_sizes else 300

        # Plot the subgraph H
        plt.figure(figsize=(int(len(path) * 1.5), int(len(path) * 1.5 * 0.8)))
        pos = nx.circular_layout(H)
        max_size = max(node_sizes) if node_sizes else 1.0
        node_colors = [size / max_size for size in node_sizes]

        nx.draw_networkx_nodes(
            H,
            pos,
            node_size=node_sizes,
            #node_color=node_colors,
            edgecolors="black",
#            cmap=plt.cm.Blues,
           # vmin=0,
           # vmax=1
        )

        arrow_len = 25 + 3 * (len(path) - 1)
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=list(H.edges()),
            arrowstyle="->",
            arrowsize=arrow_len,
            width=2,
            edge_color='red',  # highlight the path
            connectionstyle='arc3,rad=0',
            node_size=avg_node_size  # scalar value
        )


        nx.draw_networkx_labels(H, pos, font_size=10)

        plt.axis("off")
        plt.show()

    def plot_transition_matrix_most_probable_route_from_i(
            self,
            matrix,
            states,
            actions_symbols,
            start_state,  # e.g., (0, 0, 0)
            scale=1
    ):
        """
        Computes the most probable route starting from `start_state` using a greedy approach,
        and plots only the nodes and edges involved in that route.

        Parameters:
            matrix: Transition matrix (e.g., a NumPy array) of probabilities.
            states: A list of states (tuples or lists) corresponding to the rows/columns in matrix.
            actions_symbols: Mapping from state components to string symbols for labeling.
            start_state: The starting state (in the same format as in `states`).
            scale: A scaling factor applied to transition probabilities for visualization.
        """

        # 1. Prepare the graph using a deep copy of the matrix
        matrix = copy.deepcopy(matrix)
        G = nx.DiGraph()

        # Convert each state into a string label (by concatenating symbols)
        states_string = []
        for state in states:
            label = "".join(actions_symbols[i][s] for i, s in enumerate(state))
            states_string.append(label)

        # Add all states as nodes
        for s in states_string:
            G.add_node(s)

        # Tie-breaking: For each pair, keep only the dominant transition.
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i, j] > matrix[j, i]:
                    matrix[j, i] = -1
                elif matrix[i, j] < matrix[j, i]:
                    matrix[i, j] = -1

        # 2. Build the graph: add an edge if
        #    - matrix[i, j] > -1 (i.e. it survived tie-breaking)
        #    - Hamming distance between state labels is 1 (using self._hamming_dist)
        for i, s_i in enumerate(states_string):
            for j, s_j in enumerate(states_string):
                if i != j and matrix[i, j] > -1 and self._hamming_dist(s_i, s_j) == 1:
                    prob = matrix[i, j] * scale  # use scaled probability for visualization
                    G.add_edge(s_i, s_j, weight=prob)

        # 3. Convert the provided start_state into its string label
        start_label = "".join(actions_symbols[i][s] for i, s in enumerate(start_state))
        if start_label not in G.nodes:
            print(f"Start state {start_label} not recognized.")
            return

        # 4. Greedily compute the most probable route:
        # Start from start_label and at each step, choose the outgoing edge with the highest probability
        route = [start_label]
        current = start_label
        while True:
            # Get outgoing edges with data from the current node
            out_edges = list(G.out_edges(current, data=True))
            if not out_edges:
                break  # no further moves

            # Avoid cycles: filter out edges leading to nodes already in the route
            valid_edges = [edge for edge in out_edges if edge[1] not in route]
            if not valid_edges:
                break

            # Select the edge with the maximum probability ('weight')
            next_edge = max(valid_edges, key=lambda e: e[2].get('weight', 0))
            route.append(next_edge[1])
            current = next_edge[1]

        if len(route) < 2:
            print("No route found from the starting state.")
            return

        # 5. Create a subgraph H containing only the nodes and edges of the computed route
        path_edges = list(zip(route, route[1:]))
        H = nx.DiGraph()
        for node in route:
            H.add_node(node)
        for (u, v) in path_edges:
            if G.has_edge(u, v):
                H.add_edge(u, v, **G[u][v])

        # 6. Compute node sizes based on incoming edge weights (for visualization)
        in_weight_sum = {}
        for node in H.nodes():
            total_in = 0.0
            for pred in H.predecessors(node):
                edge_data = H.get_edge_data(pred, node)
                if edge_data is not None and "weight" in edge_data:
                    total_in += edge_data["weight"]
            in_weight_sum[node] = total_in
        min_size = 300
        scale_factor = 1000
        node_sizes = [1500 for node in H.nodes()]
        avg_node_size = sum(node_sizes) / len(node_sizes) if node_sizes else 300

        # 7. Plot the subgraph H (only the route)
        plt.figure(figsize=(int(len(route) * 1.5), int(len(route) * 1.5 * 0.8)))
        pos = nx.circular_layout(H)

        # Normalize node sizes for the colormap
        max_size = max(node_sizes) if node_sizes else 1.0
        node_colors = [size / max_size for size in node_sizes]

        nx.draw_networkx_nodes(
            H,
            pos,
            node_size=node_sizes,
            #node_color=node_colors,
            edgecolors="black",
            #cmap=plt.cm.Blues,
          #  vmin=0,
           # vmax=1
        )

        arrow_len = 25 + 3 * (len(route) - 1)
        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=list(H.edges()),
            arrowstyle="->",
            arrowsize=arrow_len,
            width=2,
            edge_color='red',  # highlight the route
            connectionstyle='arc3,rad=0',
            node_size=avg_node_size  # use a scalar for edge drawing
        )

        nx.draw_networkx_labels(H, pos, font_size=10)
        plt.axis("off")
        plt.show()

    @staticmethod
    def compute_stationary_distribution(matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix.T)

        idx = np.argmin(np.abs(eigenvalues - 1))

        stationary = eigenvectors[:, idx].real

        stationary = stationary / np.sum(stationary)

        return stationary

    def plot_stationary_distribution_all_pop(self, stationary_distribution, states: list[tuple[int]], actions_symbols: list[str], title: str, ylabel=None, red: [int] = None, green: [int] = None, save_file_name=None):
        fig, ax = plt.subplots(figsize=(5+len(states)//4, 4+len(states)//5))
        states_string = []
        for state in states:
            state_string = ""
            for i in range(len(state)):
                state_string += actions_symbols[i][state[i]]
            states_string.append(state_string)

        colors = []
        for label in states:
            if label in red:
                colors.append("red")
            elif label in green:
                colors.append("green")
            else:
                colors.append("gray")
        x_positions = np.arange(len(states_string))
        ax.bar(x_positions, stationary_distribution, color=colors)
        print(states_string, stationary_distribution)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(states_string, rotation=45)
        ax.set_ylim([0, 1])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(save_file_name)

        plt.show()


    def plot_stationary_distribution_per_pop(self, stationary_distribution, player:int, states: list[tuple[int]], actions_symbols: list[str], title: str, ylabel=None, save_file_name=None):
        def compute_strat_fraction(stationary_distribution, states_string, actions_symbols, player):
            strategies_fractionss = np.zeros(len(actions_symbols[player]))
            for i, st in enumerate(states_string):
                strategies_fractionss[actions_symbols[player].index(st[player])] += stationary_distribution[i]

            return strategies_fractionss

        states_string = []
        for state in states:
            state_string = ""
            for i in range(len(state)):
                state_string += actions_symbols[i][state[i]]
            states_string.append(state_string)
        colors = ["Red", "Blue", "Green", "Yellow"]
        fig, ax = plt.subplots(figsize=(5, 4))
        strategies_fractionss = compute_strat_fraction(stationary_distribution, states_string, actions_symbols, player=player)
        ax.bar(range(len(actions_symbols[player])), strategies_fractionss, color=colors[0:len(actions_symbols[player])])
        ax.set_xticks(range(len(actions_symbols[player])))
        ax.set_xticklabels(self._actions_names[player])
        ax.set_ylim([0, 1])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(save_file_name)

        plt.show()

    def plot_stationary_distributions_all_pop(self, stationary_distributions, states, plot_states, actions_symbols,
                                              save_file_name=None):
        stationary_distributions = np.array(stationary_distributions)

        states_symb = []
        for j, state in enumerate(states):
            states_symb.append("")
            for i, s in enumerate(state):
                states_symb[j] += actions_symbols[i][s]

        ind = []
        for s in plot_states:
            ind.append(states.index(s))

        plots = stationary_distributions[:, ind]
        x_labels = [f"Param {i}" for i in range(len(plots))]

        # Plot each column
        for col_idx in range(plots.shape[1]):
            plt.scatter(x_labels, plots[:, col_idx], label=f"{states_symb[ind[col_idx]]}")

        plt.xlabel("Params", fontsize=16)
        plt.ylabel("Fractions", fontsize=16)
        plt.title("Stationary Distributions All Populations", fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.ylim(0, 1)

        # Increase tick label size
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(save_file_name)
        plt.show()

    def plot_stationary_distributions_per_pop(self, stationary_distributions, states, actions_symbols, actions_names, players_names, player, save_file_name=None):
        stationary_distributions = np.array(stationary_distributions)

        states_symb = []
        for j, state in enumerate(states):
            states_symb.append("")
            for i, s in enumerate(state):
                states_symb[j] += actions_symbols[i][s]

        strategies_fractionss = np.zeros((stationary_distributions.shape[0], len(actions_symbols[player])))
        for i in range(stationary_distributions.shape[0]):
            stationary_distribution = stationary_distributions[i]
            for j, st in enumerate(states_symb):
                strategies_fractionss[i, actions_symbols[player].index(st[player])] += stationary_distribution[j]

        x_labels = [f"Param {i}" for i in range(strategies_fractionss.shape[0])]
        for col_idx in range(strategies_fractionss.shape[1]):
            plt.scatter(x_labels, strategies_fractionss[:, col_idx], label=f"{actions_names[player][col_idx]}")

        plt.xlabel("Params", fontsize=16)
        plt.ylabel("Fractions", fontsize=16)
        plt.title(f"Stationary Distributions {players_names[player]}", fontsize=18)
        plt.legend(fontsize=14)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        if save_file_name is not None:
            plt.savefig(save_file_name)

        plt.show()

    @staticmethod
    def _fermi_rule(p1_fitness: float, p2_fitness: float, beta: float) -> float:
        return 1 / (1 + math.exp(-beta * (p2_fitness - p1_fitness)))

    def moran_process(self, process, sync=False, steps: int = 50, reps: int = 1, beta: float = 1, mu: float = 0.01, return_hist: bool = False, print_rep_interval: int = None):
        if process == "bd" and sync:
            res = self._birth_death_sync(reps, steps, beta, mu, return_hist, print_rep_interval)

        elif process == "bd" and not sync:
            res = self._birth_death_async(reps, steps, beta, mu, return_hist, print_rep_interval)

        elif process == "db" and sync:
            res = self._death_birth_sync(reps, steps, beta, mu, return_hist, print_rep_interval)

        elif process == "db" and not sync:
            res = self._death_birth_async(reps, steps, beta, mu, return_hist, print_rep_interval)

        elif process == "pairwise" and not sync:
            res = self._pairwise_async(reps, steps, beta, mu, return_hist, print_rep_interval)

        elif process == "pairwise" and sync:
            res = self._pairwise_sync(reps, steps, beta, mu, return_hist, print_rep_interval)
        else:
            assert False, "Wrong process name"
        return res

    def compute_probs_of_selection(self, s, P, prev_strategies_fractionss):
        fitness = np.zeros(self._Zs[s])
        for i in range(self._Zs[s]):
            fitness[i] = self.compute_fitness(s, P[i], prev_strategies_fractionss)

        fitness_min = min(fitness)
        fitness_max = max(fitness)
        for i in range(self._Zs[s]):
            term1 = fitness[i] - fitness_min
            term2 = fitness_max - fitness_min
            if term2 == 0:
                fitness[i] = 0
            else:
                fitness[i] = term1 / term2

        fitness += 1/self._Zs[s]

        probs = fitness / np.sum(fitness)
        #print(f"prob: {probs}", f"fitness:{fitness}")
        return probs

    def _birth_death_async(self, rep: int = 1, steps: int = 50, beta: float = 1, mu: float = 0.01,
                           return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = [
            np.zeros((rep, steps + 1, len(self._actions[s])))
            for s in range(self._n)
        ]

        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[s][r, 0] = self._strategies_fractionss[s]

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [P.copy() for P in self._Ps]
            prev_strategies_fractionss = [strategies_fractions.copy() for strategies_fractions in self._strategies_fractionss]

            for t in range(steps):
                for s in range(self._n):
                    new_P = copy.deepcopy(prev_Ps[s])
                    probs = self.compute_probs_of_selection(s, new_P, prev_strategies_fractionss)
                    strat_i = np.random.choice(new_P, p=probs)
                    j = random.randint(0, len(new_P) - 1)
                    strat_j = new_P[j]

                    fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                    if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                        if random.random() < mu:
                            new_P[j] = (new_P[j] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                        else:
                            new_P[j] = strat_i

                    new_strategies_fractions = self._compute_strategies_fractions_from_P(new_P, s)
                    strategies_fractionss_hist[s][r, t + 1] = new_strategies_fractions
                    Ps_hist[s][r, t + 1] = np.array(new_P)
                    prev_Ps[s] = new_P
                    prev_strategies_fractionss[s] = new_strategies_fractions

        mean_fractionss_hist = [np.mean(hist, axis=0) for hist in strategies_fractionss_hist]

        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _birth_death_sync(self, rep: int = 1, steps: int = 50, beta: float = 1, mu: float = 0.01,
                          return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = [
            np.zeros((rep, steps + 1, len(self._actions[s])))
            for s in range(self._n)
        ]

        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[s][r, 0] = self._strategies_fractionss[s]

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")

            prev_Ps = [P.copy() for P in self._Ps]
            prev_strategies_fractionss = [strategies_fractions.copy() for strategies_fractions in self._strategies_fractionss]

            for t in range(steps):
                new_Ps = copy.deepcopy(prev_Ps)
                for s in range(self._n):
                    for i in range(len(new_Ps[s])):
                        j = random.randint(0, len(new_Ps[s]) - 1)
                        strat_i, strat_j = prev_Ps[s][i], prev_Ps[s][j]
                        fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                        fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                        if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                            if random.random() < mu:
                                new_Ps[s][j] = (new_Ps[s][j] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                            else:
                                new_Ps[s][j] = strat_i

                prev_Ps = new_Ps
                new_strategies_fractionss = self._compute_strategies_fractionss_from_Ps(new_Ps)
                prev_strategies_fractionss = new_strategies_fractionss
                for s in range(self._n):
                    strategies_fractionss_hist[s][r, t + 1] = new_strategies_fractionss[s]
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(new_Ps[s])

        mean_fractionss_hist = [np.mean(hist, axis=0) for hist in strategies_fractionss_hist]

        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _death_birth_async(self, rep: int = 1, steps: int = 50, beta: float = 1, mu: float = 0.01,
                           return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = [
            np.zeros((rep, steps + 1, len(self._actions[s])))
            for s in range(self._n)
        ]

        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[s][r, 0] = self._strategies_fractionss[s]

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [P.copy() for P in self._Ps]
            prev_strategies_fractionss = [strategies_fractions.copy() for strategies_fractions in self._strategies_fractionss]

            for t in range(steps):
                for s in range(self._n):
                    new_P = copy.deepcopy(prev_Ps[s])
                    probs = self.compute_probs_of_selection(s, new_P, prev_strategies_fractionss)

                    i = random.randint(0, len(new_P) - 1)
                    strat_j = np.random.choice(new_P, p=probs)
                    strat_i = new_P[i]

                    fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                    if random.random() < self._fermi_rule(fitness_i, fitness_j, beta):
                        if random.random() < mu:
                            new_P[i] = (new_P[i] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                        else:
                            new_P[i] = strat_j

                    new_strategies_fractions = self._compute_strategies_fractions_from_P(new_P, s)
                    strategies_fractionss_hist[s][r, t + 1] = new_strategies_fractions
                    Ps_hist[s][r, t + 1] = np.array(new_P)
                    prev_Ps[s] = new_P
                    prev_strategies_fractionss[s] = new_strategies_fractions

        mean_fractionss_hist = [np.mean(hist, axis=0) for hist in strategies_fractionss_hist]

        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _death_birth_sync(self, rep: int = 1, steps: int = 50, beta: float = 1, mu: float = 0.01,
                          return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = [
            np.zeros((rep, steps + 1, len(self._actions[s])))
            for s in range(self._n)
        ]

        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[s][r, 0] = self._strategies_fractionss[s]  # changed: state‐indexed

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [P.copy() for P in self._Ps]
            prev_strategies_fractionss = [strategies_fractions.copy() for strategies_fractions in self._strategies_fractionss]

            for t in range(steps):
                new_Ps = copy.deepcopy(prev_Ps)
                for s in range(self._n):
                    for i in range(len(new_Ps[s])):
                        j = random.randint(0, len(new_Ps[s]) - 1)
                        strat_i, strat_j = prev_Ps[s][i], prev_Ps[s][j]
                        fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                        fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)
                        if random.random() < self._fermi_rule(fitness_i, fitness_j, beta):
                            if random.random() < mu:
                                new_Ps[s][i] = (new_Ps[s][i] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                            else:
                                new_Ps[s][i] = strat_j

                prev_Ps = new_Ps
                new_strategies_fractionss = self._compute_strategies_fractionss_from_Ps(new_Ps)
                prev_strategies_fractionss = new_strategies_fractionss
                for s in range(self._n):
                    strategies_fractionss_hist[s][r, t + 1] = new_strategies_fractionss[s]
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(new_Ps[s])

        mean_fractionss_hist = [np.mean(hist, axis=0) for hist in strategies_fractionss_hist]

        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _pairwise_async(self, rep: int = 1, steps: int = 50, beta: float = 1, mu: float = 0.01,
                        return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = [
            np.zeros((rep, steps + 1, len(self._actions[s])))
            for s in range(self._n)
        ]

        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[s][r, 0] = self._strategies_fractionss[s]  # changed: state‐indexed

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [P.copy() for P in self._Ps]
            prev_strategies_fractionss = [strategies_fractions.copy() for strategies_fractions in self._strategies_fractionss]

            for t in range(steps):
                for s in range(self._n):
                    new_P = copy.deepcopy(prev_Ps[s])

                    i, j = random.sample(range(len(new_P)), 2)
                    strat_i, strat_j = new_P[i], new_P[j]
                    fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                    fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)

                    if random.random() < self._fermi_rule(fitness_i, fitness_j, beta):
                        if random.random() < mu:
                            new_P[i] = (new_P[i] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                        else:
                            new_P[i] = strat_j
                    else:
                        if random.random() < mu:
                            new_P[j] = (new_P[j] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                        else:
                            new_P[j] = strat_i

                    new_strategies_fractions = self._compute_strategies_fractions_from_P(new_P, s)
                    strategies_fractionss_hist[s][r, t + 1] = new_strategies_fractions
                    Ps_hist[s][r, t + 1] = np.array(new_P)
                    prev_Ps[s] = new_P
                    prev_strategies_fractionss[s] = new_strategies_fractions

        mean_fractionss_hist = [np.mean(hist, axis=0) for hist in strategies_fractionss_hist]

        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def _pairwise_sync(self, rep: int = 1, steps: int = 50, beta: float = 1, mu: float = 0.01,
                       return_hist: bool = False, print_rep_interval: int = None):

        strategies_fractionss_hist = [
            np.zeros((rep, steps + 1, len(self._actions[s])))
            for s in range(self._n)
        ]

        Ps_hist = [np.zeros((rep, steps + 1, self._Zs[s]), dtype=int) for s in range(self._n)]

        for r in range(rep):
            for s in range(self._n):
                Ps_hist[s][r, 0] = self._Ps[s]
                strategies_fractionss_hist[s][r, 0] = self._strategies_fractionss[s]  # changed: state‐indexed

        for r in range(rep):
            if (print_rep_interval is not None) and (r >= print_rep_interval) and (r % print_rep_interval == 0):
                print(f"Replicate: {r}")
            prev_Ps = [P.copy() for P in self._Ps]
            prev_strategies_fractionss = [strategies_fractions.copy() for strategies_fractions in self._strategies_fractionss]

            for t in range(steps):
                new_Ps = copy.deepcopy(prev_Ps)
                for s in range(self._n):
                    for i in range(len(new_Ps[s])):
                        j = random.randint(0, len(new_Ps[s]) - 1)
                        strat_i, strat_j = prev_Ps[s][i], prev_Ps[s][j]
                        fitness_i = self.compute_fitness(s, strat_i, prev_strategies_fractionss)
                        fitness_j = self.compute_fitness(s, strat_j, prev_strategies_fractionss)

                        if random.random() < self._fermi_rule(fitness_j, fitness_i, beta):
                            if random.random() < mu:
                                new_Ps[s][j] = (new_Ps[s][j] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                            else:
                                new_Ps[s][j] = strat_i
                        else:
                            if random.random() < mu:
                                new_Ps[s][i] = (new_Ps[s][i] + random.randint(0, len(self._actions[s]) - 1)) % len(self._actions[s])
                            else:
                                new_Ps[s][i] = strat_j

                prev_Ps = new_Ps
                new_strategies_fractionss = self._compute_strategies_fractionss_from_Ps(new_Ps)
                prev_strategies_fractionss = new_strategies_fractionss
                for s in range(self._n):
                    strategies_fractionss_hist[s][r, t + 1] = new_strategies_fractionss[s]  # changed: state‐indexed
                for s in range(self._n):
                    Ps_hist[s][r, t + 1] = np.array(new_Ps[s])

        mean_fractionss_hist = [np.mean(hist, axis=0) for hist in strategies_fractionss_hist]

        if return_hist:
            return mean_fractionss_hist, strategies_fractionss_hist, Ps_hist
        return mean_fractionss_hist

    def plot_strategy_evol(self, strategies_fractionss_hist, actions: [int], xlabel: str = None,
                       ylabel: str = None, title: str = None):
        strategies_fractionss_hist = [strategies_fractions_hist[:, actions[i]] for i, strategies_fractions_hist in enumerate(strategies_fractionss_hist)]
        timesteps = np.arange(len(strategies_fractionss_hist[0]))
        plt.figure(figsize=(10, 6))
        for col in range(self._n):
            plt.plot(timesteps, strategies_fractionss_hist[col], label=self._players_names[col] + " Strategy: " + str(self._actions_names[col][actions[col]]))
        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_stationary_dist(self, Ps_hist,  player: int, action: int, xlabel: str = None,
                                  ylabel: str = None, title: str = None):
        P_hist = Ps_hist[player]
        states = np.sum(P_hist == action, axis=2)

        unique_states, counts = np.unique(states, return_counts=True)
        fractions = counts / sum(counts)
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
        strategies_countss = []
        for i, pop in enumerate(strategies_fractionss):
            strategies_countss.append([])
            for frac in pop:
                strategies_countss[i].append(int(self._Zs[i] * frac))
            while sum(strategies_countss[i]) < self._Zs[i]:
                m = len(strategies_countss[i]) - 1
                strategies_countss[i][random.randint(0, m)] += 1
            while sum(strategies_countss[i]) > self._Zs[i]:
                m = len(strategies_countss[i]) - 1
                strategies_countss[i][random.randint(0, m)] -= 1
        return strategies_countss

    def _compute_strategies_countss_from_Ps(self, Ps):
        strategies_countss = []

        for i, P in enumerate(Ps):
            counts = [0] * len(self._actions[i])
            for strat in P:
                counts[strat] += 1
            strategies_countss.append(counts)

        return strategies_countss

    def _compute_strategies_counts_from_P(self, P, s):
        strategies_counts = [0] * len(self._actions[s])
        for strat in P:
            strategies_counts[strat] += 1

        return strategies_counts

    def _compute_strategies_fractionss_from_countss(self, strategies_countss):
        strategies_fractionss = []
        for i, pop in enumerate(strategies_countss):
            strategies_fractionss.append([])
            for count in pop:
                strategies_fractionss[i].append(round(float(count / self._Zs[i]), 4))
            if sum(strategies_fractionss[i]) != 1:
                m = random.randint(0, len(strategies_countss[i]) - 1)
                strategies_fractionss[i][m] = round(strategies_fractionss[i][m] + 1 - sum(strategies_fractionss[i]), 4)

        return strategies_fractionss

    def _compute_strategies_fractions_from_counts(self, strategies_counts, s):
        strategies_fractions = []
        for count in strategies_counts:
            strategies_fractions.append(round(float(count / self._Zs[s]), 4))
        if sum(strategies_fractions) != 1:
            m = random.randint(0, len(strategies_counts) - 1)
            strategies_fractions[m] = round(strategies_fractions[m] + 1 - sum(strategies_fractions), 4)

        return strategies_fractions

    def _compute_strategies_fractionss_from_Ps(self, Ps):
        strategies_countss = self._compute_strategies_countss_from_Ps(Ps)
        strategies_fractionss = self._compute_strategies_fractionss_from_countss(strategies_countss)
        return strategies_fractionss

    def _compute_strategies_fractions_from_P(self, P, s):
        strategies_counts = self._compute_strategies_counts_from_P(P, s)
        strategies_fractionss = self._compute_strategies_fractions_from_counts(strategies_counts, s)
        return strategies_fractionss
    # Setters

    def compute_gradient(self, config, beta, mu):
        gradients = []
        strategies_fractionss = []
        for i in range(self._n):
            j_i = config[i]
            strategies_fractionss.append([(self._Zs[i] - j_i) / self._Zs[i], j_i / self._Zs[i]])

        for i in range(self._n):
            f_C = self.compute_fitness(i, 1, strategies_fractionss)
            f_D = self.compute_fitness(i, 0, strategies_fractionss)
            Z = self._Zs[i]
            j = config[i]
            if Z > 1:
                term1 = ((Z - j) / Z) * (j / (Z - 1))
            else:
                term1 = 0

            term2 = ((Z - 2 * j) / Z) * mu

            # The hyperbolic tangent ensures the term scales between -1 and 1.
            gradient_i = term1 * math.tanh(beta * (f_C - f_D) / 2) + term2
            gradients.append(gradient_i)

        return gradients

    def compute_gradient_of_selection(self, beta, mu):
        if self._n != 3:
            print("plot_gradient_field is implemented for 3 populations only.")
            return

        Z1, Z2, Z3 = self._Zs[0], self._Zs[1], self._Zs[2]
        X, Y, Z = [], [], []
        G1, G2, G3 = [], [], []

        # Iterate over all possible configurations: j1 from 0 to Z1, j2 from 0 to Z2, j3 from 0 to Z3.
        for j1 in range(Z1 + 1):
            for j2 in range(Z2 + 1):
                for j3 in range(Z3 + 1):
                    config = [j1, j2, j3]

                    gradient = self.compute_gradient(config, beta, mu)
                    # Convert counts to fractions.
                    x = (j1) / Z1
                    y = (j2) / Z2
                    z = (j3) / Z3

                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    G1.append(gradient[0])
                    G2.append(gradient[1])
                    G3.append(gradient[2])

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        G1 = np.array(G1)
        G2 = np.array(G2)
        G3 = np.array(G3)
        return X, Y, Z, G1, G2, G3

    def plot_gradient_of_selection(self, X, Y, Z, G1, G2, G3, title=None, players_names=None, fraction_name=None, legend=None, threshold=0.02):
        # Compute the magnitude of the gradient for each configuration.
        mag = np.sqrt(G1 ** 2 + G2 ** 2 + G3 ** 2)
        # Normalize magnitudes for the colormap.
        norm = plt.Normalize(vmin=mag.min(), vmax=mag.max())
        # Use a colormap that goes from blue (low) to red (high). You can choose another cmap if desired.
        colors = plt.cm.jet(norm(mag))

        # Create a 3D quiver plot.
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot arrows: each arrow is located at (X, Y, Z) with components (G1, G2, G3)
        q = ax.quiver(X, Y, Z, G1, G2, G3, length=0.05, normalize=True, color=colors)

        ax.set_xlabel(f"Fraction of {fraction_name} in {players_names[0]}")
        ax.set_ylabel(f"Fraction of {fraction_name} in {players_names[1]}")
        ax.set_zlabel(f"Fraction of {fraction_name} in {players_names[2]}")
        ax.set_title(title)

        # Create a ScalarMappable for the colorbar using the same colormap and normalization.
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])  # Dummy array for the ScalarMappable.
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label(legend)

        mask = (mag < threshold)
        if np.any(mask):
            # Plot them as black spheres or any style you prefer
            ax.scatter(
                X[mask], Y[mask], Z[mask],
                c='k', s=40, marker='o',
                label=f"Near zero gradient (<{threshold})"
            )
            # Show a legend entry
            ax.legend()

        plt.show()

    def compute_iterative_replicator_dynamics(self, dt, steps):
        strategies_fractionss = np.array(self._strategies_fractionss)

        strategies_fractionss_hist = np.zeros((steps+1, self._n, len(self._actions)))
        gradients = np.zeros((steps+1, self._n, len(self._actions)))
        for i in range(self._n):
            strategies_fractionss_hist[0, i] = strategies_fractionss[i].copy()
            gradients[0, i] = np.array([0 for k in range(len(self._actions))])

        for s in range(1, steps+1):
            for i in range(self._n):
                fitnesses = np.array([self.compute_fitness(i, strat, strategies_fractionss) for strat in range(len(self._actions))])
                avg_fitness = np.dot(strategies_fractionss[i, :], fitnesses)

                for k in range(len(self._actions)):
                    x_dot = strategies_fractionss[i, k] * (fitnesses[k] - avg_fitness)
                    strategies_fractionss[i, k] = strategies_fractionss[i, k] + dt * x_dot
                    gradients[s, i, k] = x_dot

                total = np.sum(strategies_fractionss[i, :])
                if total > 0:
                    strategies_fractionss[i, :] = strategies_fractionss[i, :] / total
                else:
                    strategies_fractionss[i, :] = 1.0 / len(self._actions)

            strategies_fractionss_hist[s] = strategies_fractionss.copy()

        return strategies_fractionss_hist, gradients

    def plot_replicator_dynamics_strategies_evol(self, strategies_fractionss_hist):
        plt.figure(figsize=(8, 5))

        for player in range(len(self._players_names)):
            for strat_idx in range(len(self._actions_names)):
                plt.plot(range(len(strategies_fractionss_hist[1:, 0, strat_idx])), strategies_fractionss_hist[1:, 0, strat_idx],
                         label=f"Strategy {self._actions_names[strat_idx]}")
            plt.xlabel("Time")
            plt.ylabel("Strategy Fraction")
            plt.title(f"Replicator Dynamics {self._players_names[player]}")
            plt.legend()
            plt.show()

    def plot_replicator_dynamics_gradient_evol(self, strategies_fractionss_hist, gradient):
        plt.figure(figsize=(8, 5))

        for player in range(len(self._players_names)):
            for strat_idx in range(len(self._actions_names)):
                plt.plot(strategies_fractionss_hist[1:, 0, strat_idx], gradient[1:, 0, strat_idx],
                         label=f"Strategy {self._actions_names[strat_idx]}")
                plt.xlabel("Strategy Fraction")
                plt.ylabel("Gradient of Selection")
                plt.title(f"Replicator Dynamics {self._players_names[player]}")
                plt.legend()
                plt.show()

    def set_strategies_counts(self, new_strategies_counts):
        self._strategies_countss = copy.deepcopy(new_strategies_counts)
        self._strategies_fractionss = self._compute_strategies_fractionss_from_countss(self._strategies_countss)

    def set_strategies_fractionss(self, new_strategies_fractionss):
        self._strategies_fractionss = copy.deepcopy(new_strategies_fractionss)
        self._strategies_countss = self._count_strategies_from_fractions(self._strategies_countss)

    def set_players_names(self, new_players_names):
        assert self._parse_players_names(new_players_names), "Invalid players names"
        self._players_names = copy.deepcopy(new_players_names)

    def set_actions_names(self, new_actions_names):
        assert self._parse_actions_names(new_actions_names), "Invalid actions names"
        self._actions_names = copy.deepcopy(new_actions_names)


if __name__ == "__main__":
    pass