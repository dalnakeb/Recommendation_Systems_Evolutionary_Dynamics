"""import copy
import math

import numpy as np
import random
from itertools import product
import matplotlib.pyplot as plt


class Game:
    def __init__(self, Ns: list[int], fractionss: list[list[float]], payoff_matrix=None):
        self._Ps = None
        self._payoff_matrix = None
        self._Ns = Ns
        self._n = len(self._Ns)
        self._fractionss = fractionss
        self._strategies = [i for i in range(len(self._fractionss[0]))]

        self._init_populations()

        if payoff_matrix is None:
            self._init_payoff_matrix()
        else:
            assert self._parse_payoff_matrix(payoff_matrix)
            self._payoff_matrix = payoff_matrix

    def _parse_payoff_matrix(self, payoff_matrix):
        if not type(payoff_matrix) is dict:
            return False
        if len(payoff_matrix) == 2**self._n:
            return False

        for value in payoff_matrix.values():
            if not type(value) is list and len(value) != self._n and not all([type(x) is int for x in value]):
                return False
        for key in payoff_matrix.keys():
            if not type(key) is list and len(key) != self._n and not all([type(x) is int for x in key]):
                return False
        return True

    def _init_population(self, N: int, fractions: list[float]) -> np.array:
        P = np.zeros(N, dtype=int)
        current_ind = 0
        for fraction_ind in range(0, len(fractions)):
            P[current_ind:int(current_ind + int(N*fractions[fraction_ind]))] = self._strategies[fraction_ind]
            current_ind += int(N*fractions[fraction_ind])
        np.random.shuffle(P)
        return list(P)

    def _init_populations(self):
        self._Ps = []
        for N, fractions in zip(self._Ns, self._fractionss):
            self._Ps.append(self._init_population(N, fractions))

    def _init_payoff_matrix(self):
        strat_combinations = list(product(self._strategies, repeat=self._n))
        self._payoff_matrix = {}
        for strat_combination in strat_combinations:
            self._payoff_matrix[strat_combination] = [0 for _ in range(self._n)]

    def compute_fitness(self, x, y, z, X, sector):
        f = 0
        P = self._payoff_matrix
        if str.lower(sector) == "public":
            f = (y*z*P[(X, 1, 1)][0] +
                 (1-y)*z*P[(X, 0, 1)][0] +
                 y*(1-z)*P[(X, 1, 0)][0] +
                 (1-y)*(1-z)*P[(X, 0, 0)][0])
        elif str.lower(sector) == "private":
            f = (x*z*P[(1, X, 1)][1] +
                 (1-x)*z*P[(0, X, 1)][1] +
                 x*(1-z)*P[(1, X, 0)][1] +
                 (1-x)*(1-z)*P[(0, X, 0)][1])
        elif str.lower(sector) == "civil":
            f = (x*y*P[(1, 1, X)][2] +
                 (1-x)*y*P[(0, 1, X)][2] +
                 x*(1-y)*P[(1, 0, X)][2] +
                 (1-x)*(1-y)*P[(0, 0, X)][2])

        return f

    def _fermi_rule(self, p1_fitness, p2_fitness, beta):
        return 1/(1 + math.exp(-beta*(p2_fitness - p1_fitness)))

    def birth_death(self, rep=1, steps=50, beta=10, u=0.1, return_hist=False, print_rep_count_interval=None):
        sectors = ["public", "private", "civil"]
        fractionss_hist = np.zeros((rep, steps+1, 3, 2))
        Ps_hist = [np.zeros((rep, steps+1, self._Ns[0])), np.zeros((rep, steps+1, self._Ns[1])), np.zeros((rep, steps+1, self._Ns[2]))]
        for r in range(rep):
            if print_rep_count_interval is not None and r % print_rep_count_interval == 0:
                print(f"reps: {r}")
            prev_Ps = copy.deepcopy(self._Ps)
            prev_fractionss = copy.deepcopy(self._fractionss)
            fractionss_hist[r][0] = prev_fractionss
            Ps_hist[0][r, 0] = np.array(prev_Ps[0])
            Ps_hist[1][r, 0] = np.array(prev_Ps[1])
            Ps_hist[2][r, 0] = np.array(prev_Ps[2])

            for i in range(steps):
                for sect in range(3):
                    p1_ind = -1
                    p2_ind = -1
                    while p1_ind == p2_ind:
                        p1_ind = random.randint(0, len(prev_Ps[sect]) - 1)
                        p2_ind = random.randint(0, len(prev_Ps[sect]) - 1)

                    p1 = prev_Ps[sect][p1_ind]
                    p2 = prev_Ps[sect][p2_ind]
                    p1_fitness = self.compute_fitness(prev_fractionss[0][1],
                                                      prev_fractionss[1][1],
                                                      prev_fractionss[2][1],
                                                      p1,
                                                      sectors[sect])

                    p2_fitness = self.compute_fitness(prev_fractionss[0][1],
                                                      prev_fractionss[1][1],
                                                      prev_fractionss[2][1],
                                                      p2,
                                                      sectors[sect])

                    prob = self._fermi_rule(p1_fitness, p2_fitness, beta)
                    imit_test = random.random()
                    mut_test = random.random()
                    if imit_test < prob:
                        prev_Ps[sect][p1_ind] = prev_Ps[sect][p2_ind]

                    p1_ind = random.randint(0, len(prev_Ps[sect]) - 1)

                    if mut_test < u:
                        prev_Ps[sect][p1_ind] = (prev_Ps[sect][p1_ind] + 1) % 2

                    prev_fractionss[0][1] = np.sum(np.array(prev_Ps[0]) == 1)/self._Ns[0]
                    prev_fractionss[0][0] = 1-prev_fractionss[0][1]

                    prev_fractionss[1][1] = np.sum(np.array(prev_Ps[1]) == 1)/self._Ns[1]
                    prev_fractionss[1][0] = 1-prev_fractionss[1][1]

                    prev_fractionss[2][1] = np.sum(np.array(prev_Ps[2]) == 1)/self._Ns[2]
                    prev_fractionss[2][0] = 1-prev_fractionss[2][1]

                Ps_hist[0][r, i+1] = np.array(prev_Ps[0])
                Ps_hist[1][r, i+1] = np.array(prev_Ps[1])
                Ps_hist[2][r, i+1] = np.array(prev_Ps[2])
                fractionss_hist[r][i+1] = np.array(prev_fractionss)

        mean_fractionss_hist = np.mean(fractionss_hist, axis=0)
        if return_hist:
            return mean_fractionss_hist, fractionss_hist, Ps_hist

        return mean_fractionss_hist

    def visualize_evol(self, fractionss_hist, players: list[str], xlabel=None, ylabel=None, title=None):
        timesteps = np.arange(fractionss_hist.shape[0])

        plt.figure(figsize=(10, 6))
        if len(fractionss_hist.shape) > 1:
            for col in range(fractionss_hist.shape[1]):
                plt.plot(timesteps, fractionss_hist[:, col], label=f"{players[col]}")

        else:
            plt.plot(timesteps, fractionss_hist[:], label=f"{players[0]}")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_stationary_dist(self, Ps_hist, title=None):
        states = Ps_hist.sum(axis=2).flatten().astype(int)
        dist, frac = np.unique(states, return_counts=True)
        frac = frac/len(states)
        max_frac = max(frac)
        max_frac_ind = int(np.argmax(frac))
        plt.plot(dist, frac)
        plt.scatter(dist[max_frac_ind], max_frac, color="red", label=f"Most Common State ({dist[max_frac_ind]}, {round(max_frac, 4)})")
        plt.xlabel("State")
        plt.xlabel("Fraction")
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()

    def set_payoff_matrix(self, payoff_matrix):
        self._payoff_matrix = payoff_matrix

    def get_payoff_matrix(self):
        return self._payoff_matrix

    def get_populations(self):
        return self._Ps


if __name__ == "__main__":
    pass"""

import math
import random
from itertools import product
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import copy

import numpy as np
import matplotlib.pyplot as plt


class Game:
    def __init__(self, Ns: list[int], fractionss: list[list[float]], payoff_matrix=None):
        self._Ns = Ns  # Population sizes for each sector
        self._n = len(Ns)  # Number of sectors
        self._fractionss = copy.deepcopy(fractionss)
        # Assume binary strategies; derive strategies from the first fraction list length.
        self._strategies = list(range(len(fractionss[0])))

        # Initialize populations for each sector.
        self._init_populations()

        if payoff_matrix is None:
            self._init_payoff_matrix()
        else:
            assert self._parse_payoff_matrix(payoff_matrix), "Invalid payoff matrix format."
            self._payoff_matrix = payoff_matrix

    def _parse_payoff_matrix(self, payoff_matrix: dict) -> bool:
        if not isinstance(payoff_matrix, dict):
            return False
        # Retain the original condition (though note it is a bit unusual).
        if len(payoff_matrix) == 2 ** self._n:
            return False

        # Validate keys and values.
        for key, value in payoff_matrix.items():
            if not (isinstance(key, list) and len(key) == self._n and all(isinstance(x, int) for x in key)):
                return False
            if not (isinstance(value, list) and len(value) == self._n and all(isinstance(x, int) for x in value)):
                return False
        return True

    def _init_population(self, N: int, fractions: list[float]) -> list[int]:
        """Initialize a single population based on the provided fractions."""
        population = []
        for strategy, frac in enumerate(fractions):
            count = int(N * frac)
            population.extend([strategy] * count)
        # In case of rounding issues, adjust the length to exactly N.
        while len(population) < N:
            population.append(self._strategies[0])
        population = population[:N]
        random.shuffle(population)
        return population

    def _init_populations(self):
        """Initialize populations for all sectors."""
        self._Ps = [self._init_population(N, fractions) for N, fractions in zip(self._Ns, self._fractionss)]

    def _init_payoff_matrix(self):
        """Create a default payoff matrix with zero payoffs."""
        strat_combinations = list(product(self._strategies, repeat=self._n))
        self._payoff_matrix = {tuple(combo): [0] * self._n for combo in strat_combinations}

    def compute_fitness(self, x: float, y: float, z: float, strategy: int, sector: str) -> float:
        """
        Compute fitness for a given strategy in the specified sector using the payoff matrix.
        x, y, z are the fraction of strategy 1 in the three sectors: public, private, and civil.
        """
        P = self._payoff_matrix
        sector_lower = sector.lower()
        if sector_lower == "public":
            return (y * z * P[(strategy, 1, 1)][0] +
                    (1 - y) * z * P[(strategy, 0, 1)][0] +
                    y * (1 - z) * P[(strategy, 1, 0)][0] +
                    (1 - y) * (1 - z) * P[(strategy, 0, 0)][0])
        elif sector_lower == "private":
            return (x * z * P[(1, strategy, 1)][1] +
                    (1 - x) * z * P[(0, strategy, 1)][1] +
                    x * (1 - z) * P[(1, strategy, 0)][1] +
                    (1 - x) * (1 - z) * P[(0, strategy, 0)][1])
        elif sector_lower == "civil":
            return (x * y * P[(1, 1, strategy)][2] +
                    (1 - x) * y * P[(0, 1, strategy)][2] +
                    x * (1 - y) * P[(1, 0, strategy)][2] +
                    (1 - x) * (1 - y) * P[(0, 0, strategy)][2])
        else:
            raise ValueError("Invalid sector name provided.")

    def _fixation_prob(self, i, j, sector, beta, Ps_fracs):
        sectors = ["public", "private", "civil"]
        sect_ind = sectors.index(str.lower(sector))
        p_ij = 0
        N = self._Ns[sect_ind]
        for l in range(1, N):
            p = 1
            for k in range(1, l+1):
                x, y, z = 0, 0, 0
                if str.lower(sector) == "public":
                    x = k/N
                    y = Ps_fracs[1]
                    z = Ps_fracs[2]
                elif str.lower(sector) == "private":
                    x = Ps_fracs[0]
                    y = k / N
                    z = Ps_fracs[2]
                elif str.lower(sector) == "civil":
                    x = Ps_fracs[0]
                    y = Ps_fracs[1]
                    z = k / N

                p1_fitness = self.compute_fitness(x, y, z, i, sector)
                p2_fitness = self.compute_fitness(x, y, z, j, sector)
                T_k_minus = k/N * (N - k)/N * self._fermi_rule(p1_fitness, p2_fitness,beta)
                T_k_plus = k/N * (N - k)/N * self._fermi_rule(p2_fitness, p1_fitness,beta)
                p *= T_k_minus/T_k_plus
            p_ij += p
        p_ij = 1/(1+p_ij)
        return p_ij

    def _compute_trans_matrix(self, beta):
        """
            Compute the 8x8 transition matrix (m) for the monomorphic states.

            The state space is given by all triplets in {0,1}^3 representing the strategy in
            the public, private, and civil sectors respectively.

            For any two states i and j that differ in exactly one sector, the transition probability
            from state i to j is given by:
                 Λ_ij = ρ_ij / 3,
            where ρ_ij is computed using the _fixation_prob function for the sector that differs.

            Diagonal elements are set so that each row sums to 1.

            Parameters:
                beta (float): Selection intensity used in the fixation probability computation.

            Returns:
                m (np.ndarray): An 8x8 numpy array representing the transition matrix.
            """
        # Define the mapping from coordinate index to sector name.
        sector_names = {0: "public", 1: "private", 2: "civil"}

        # Create the list of all monomorphic states: each state is a tuple (s0, s1, s2) with s in {0, 1}.
        states = list(product([0, 1], repeat=3))
        num_states = len(states)  # Should be 8.

        # Initialize the transition matrix with zeros.
        m = np.zeros((num_states, num_states))

        # Loop over all pairs of states.
        for i in range(num_states):
            state_i = states[i]  # Current (resident) state.
            # For each possible target state j:
            for j in range(num_states):
                # Skip self-transitions here; we'll fill the diagonal later.
                if i == j:
                    continue

                state_j = states[j]
                # Determine in how many sectors the states differ.
                differences = [idx for idx in range(3) if state_i[idx] != state_j[idx]]

                # Only single-sector differences are accessible via one mutation and fixation.
                if len(differences) == 1:
                    sector_idx = differences[0]
                    sector = sector_names[sector_idx]
                    # For the focal sector, the resident strategy is state_i[sector_idx] and the mutant is state_j[sector_idx].
                    # For the other sectors, the fractions are taken from state_i.
                    Ps_fracs = [float(state_i[0]), float(state_i[1]), float(state_i[2])]

                    # Compute the fixation probability using your provided function.
                    # Note: _fixation_prob expects the resident strategy, the mutant strategy,
                    # the sector, beta, and the fractions for the other sectors (as explained above).
                    fixation_prob = self._fixation_prob(state_i[sector_idx], state_j[sector_idx], sector, beta,
                                                        Ps_fracs)

                    # Since a mutation could occur in any of the three sectors with equal chance,
                    # divide the fixation probability by 3.
                    m[i, j] = fixation_prob / 3.0

        # Set the diagonal elements so that each row sums to 1.
        for i in range(num_states):
            m[i, i] = 1 - np.sum(m[i, :])

        return m

    def plot_transition_matrix(self, matrix):
        """
        Plots an 8x8 transition matrix as a 3D cube of nodes (DDD, DDC, DCD, DCC,
        CDD, CDC, CCD, CCC). Only edges between adjacent corners (Hamming distance = 1)
        are drawn. Each edge is labeled with its transition probability if above `threshold`.

        Args:
            matrix (np.ndarray): 8x8 transition matrix (row-stochastic).
                                 Row i -> Column j = Probability of transitioning
                                 from state i to state j.
            state_labels (list): Optional list of 8 labels in the exact row/column order
                                 of `matrix`. Default:
                                   ["DDD","DDC","DCD","DCC","CDD","CDC","CCD","CCC"]
            threshold (float):   Only edges with probability > threshold are shown.
        """
        # Default ordering that matches corners of the cube:
        # 0 -> DDD, 1 -> DDC, 2 -> DCD, 3 -> DCC,
        # 4 -> CDD, 5 -> CDC, 6 -> CCD, 7 -> CCC
        state_labels = ["DDD", "DDC", "DCD", "DCC", "CDD", "CDC", "CCD", "CCC"]
        threshold = 0.06
        # Map each label to a (x,y,z) corner in the unit cube
        # Adjust if your matrix's row order is different
        pos_3d = {
            "DDD": (0, 0, 0),
            "DDC": (0, 0, 1),
            "DCD": (0, 1, 0),
            "DCC": (0, 1, 1),
            "CDD": (1, 0, 0),
            "CDC": (1, 0, 1),
            "CCD": (1, 1, 0),
            "CCC": (1, 1, 1),
        }

        # Create a figure and 3D axes
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111, projection='3d')

        # Optional: give a title (remove or comment out if you prefer no title)
        # ax.set_title("3D Cube - Only Adjacent Transitions", pad=15)

        # Plot each state as a node
        for idx, label in enumerate(state_labels):
            x, y, z = pos_3d[label]
            # Example: highlight CCC with a bigger marker
            if label == "CCC":
                ax.scatter(x, y, z, color='darkgray', s=250, edgecolors='black')
            else:
                ax.scatter(x, y, z, color='lightgray', s=120, edgecolors='black')

            # Add text label at the same coordinate
            ax.text(x, y, z, label, fontsize=10, ha='center', va='center', zorder=10)

        # Function to compute Hamming distance between two 3-character strings
        # (e.g., "DDD" and "DDC" differ in 1 position)
        def hamming_distance(s1, s2):
            return sum(a != b for a, b in zip(s1, s2))

        # Draw edges only for states differing by exactly one coordinate
        for i, from_label in enumerate(state_labels):
            x1, y1, z1 = pos_3d[from_label]
            for j, to_label in enumerate(state_labels):
                if i == j:
                    continue
                p_ij = matrix[i, j]

                # Only connect if states differ in exactly 1 coordinate
                if hamming_distance(from_label, to_label) == 1 and p_ij > threshold:
                    x2, y2, z2 = pos_3d[to_label]

                    # Draw a line (edge) from (x1,y1,z1) to (x2,y2,z2)
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color='black', alpha=0.8)

                    # Label the edge near the midpoint
                    mx, my, mz = (x1 + x2)/2, (y1 + y2)/2, (z1 + z2)/2
                    ax.text(mx, my, mz, f"{p_ij:.2f}", color='red', fontsize=8)

        # ----------------- Remove Background -----------------
        # Turn off axis planes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # transparent background
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Hide axis lines / bounding box
        ax._axis3don = False

        # Optionally, you can keep a little margin or set limits exactly
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 1.1)

        plt.show()

    @staticmethod
    def _fermi_rule(p1_fitness: float, p2_fitness: float, beta: float) -> float:
        """Return the probability of imitation using the Fermi rule."""
        return 1 / (1 + math.exp(-beta * (p2_fitness - p1_fitness)))

    def birth_death(self, rep: int = 1, steps: int = 50, beta: float = 10, u: float = 0.1,
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
        sectors = ["public", "private", "civil"]
        num_sectors = len(sectors)
        # History arrays: fractionss_hist shape: (rep, steps+1, sectors, 2 strategies)
        fractionss_hist = np.zeros((rep, steps + 1, num_sectors, 2))
        # Ps_hist for each sector: shape: (rep, steps+1, population size)
        Ps_hist = [np.zeros((rep, steps + 1, self._Ns[s]), dtype=int) for s in range(num_sectors)]

        for r in range(rep):
            if print_rep_interval is not None and r % print_rep_interval == 0:
                print(f"Replicate: {r}")
            # Use shallow copies since populations and fractions are lists of primitives.
            prev_Ps = [pop.copy() for pop in self._Ps]
            prev_fractionss = [fracs.copy() for fracs in self._fractionss]

            # Record initial state.
            for s in range(num_sectors):
                Ps_hist[s][r, 0] = np.array(prev_Ps[s])
                fractionss_hist[r, 0, s] = np.array(prev_fractionss[s])

            for t in range(steps):
                # Process each sector.
                for s in range(num_sectors):
                    population = prev_Ps[s]
                    pop_size = len(population)

                    # Imitation: select two distinct individuals.
                    i, j = random.sample(range(pop_size), 2)
                    strat_i, strat_j = population[i], population[j]
                    # Compute fitness using current fractions (x, y, z correspond to sectors 0, 1, 2 respectively).
                    fitness_i = self.compute_fitness(
                        prev_fractionss[0][1],
                        prev_fractionss[1][1],
                        prev_fractionss[2][1],
                        strat_i,
                        sectors[s]
                    )
                    fitness_j = self.compute_fitness(
                        prev_fractionss[0][1],
                        prev_fractionss[1][1],
                        prev_fractionss[2][1],
                        strat_j,
                        sectors[s]
                    )
                    if random.random() < self._fermi_rule(fitness_i, fitness_j, beta):
                        population[i] = strat_j

                    # Mutation: select one individual randomly and flip its strategy with probability u.
                    mut_index = random.randrange(pop_size)
                    if random.random() < u:
                        population[mut_index] = (population[mut_index] + 1) % 2

                    # Update fraction for the current sector (assuming binary strategies: 0 and 1).
                    fraction_strategy1 = np.sum(np.array(population) == 1) / self._Ns[s]
                    prev_fractionss[s][1] = fraction_strategy1
                    prev_fractionss[s][0] = 1 - fraction_strategy1

                # Record state after processing all sectors.
                for s in range(num_sectors):
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
        self._payoff_matrix = payoff_matrix

    def get_payoff_matrix(self) -> dict:
        return self._payoff_matrix

    def get_populations(self) -> list:
        return self._Ps


if __name__ == "__main__":
    pass