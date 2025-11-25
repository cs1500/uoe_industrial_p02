# code originally written by Adele
# modified by cs

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

try:
    del SchellingParams
except NameError:
    pass
    
@dataclass
class SchellingParams:
    """
    Helper class to initialise Schelling model parameters.
    """
    # grid size values
    width: int = 50
    height: int = 50

    # fraction of cells empty
    empty_ratio: float = 0.12

    # utility function
    similarity_weight: float = 1.0
    affordability_weight: float = 1.0
    utility_threshold: float = 0.0 # tau

    n_iterations: int = 250
    income_low: float = 80.0
    income_high: float = 120.0
    seed: int = 666

class SchellingModel:
    """
    Schelling model, modified.

    Parameters
    ----------
    - p: SchellingParams class, with model parameters
    - price_grid: 2d Numpy array with initial housing prices V^t(x_1,x_2),
        where t=0. If not supplied, automatically generated!
    """
    def __init__(self, p: SchellingParams, price_grid: Optional[np.ndarray] = None):
        self.p = p # stores model parameters
        self.rng = np.random.default_rng(p.seed) # set seed
        H, W = p.height, p.width

        n_cells = H * W
        n_empty = int(round(p.empty_ratio * n_cells)) # number of empty cells
        n_agents = n_cells - n_empty

        # create cultural groups in agents (-1 and +1)
        half = n_agents // 2
        agents = np.array( # around half for each cultural group
            [-1]*half + [1]*(n_agents - half) + [0]*n_empty, dtype=int
        )
        self.rng.shuffle(agents) # then randomly permute in grid
        self.agent_grid = agents.reshape(H, W)

        if price_grid is None:
            # default initial housing price generation
            # houses in one diagonal will be more expensive
            x = np.linspace(-1, 1, W)
            y = np.linspace(-1, 1, H)
            X, Y = np.meshgrid(x, y)
            base = 100 + 10*X + 5*Y  
            noise = self.rng.normal(0, 2.0, size=(H, W))
            self.price_grid = base + noise
        else:
            #assert price_grid.shape == (H, W),
            assert price_grid.shape == (H, W)
            self.price_grid = price_grid.astype(float)

        # generate random incomes for agents
        incomes = self.rng.uniform(p.income_low, p.income_high, size=n_agents)
        # pack incomes into a grid aligned with agent cells (-1/+1), 0 for empty
        self.income_grid = np.zeros((H, W), dtype=float)
        idx_agents = np.argwhere(self.agent_grid != 0) # agent occupied cells
        self.rng.shuffle(idx_agents)
        for k, (i, j) in enumerate(idx_agents):
            if k >= incomes.size: break
            self.income_grid[i, j] = incomes[k]

        self.scale = max(1e-9, float(self.price_grid.max() - self.price_grid.min()))

        self.history_agents: List[np.ndarray] = [self.agent_grid.copy()]
        self.history_utility: List[np.ndarray] = []
        self.moves_per_iter: List[int] = []

    @staticmethod
    def neighbours_moore(arr: np.ndarray, i: int, j: int) -> np.ndarray:
        """
        Return the Moore neighbourhood (r=1) of a cell (i,j) in supplied 2d
        array. Needs to be extended to arbitrary neighbourhood r.

        Returns
        -------
        - np.ndarray: 1d array containing values of neighbouring cells
        """
        H, W = arr.shape
        i0, i1 = max(0, i-1), min(H-1, i+1)
        j0, j1 = max(0, j-1), min(W-1, j+1)
        block = arr[i0:i1+1, j0:j1+1]
        out = block.flatten()
        # remove center
        idx_center = (i - i0) * (j1 - j0 + 1) + (j - j0)
        return np.delete(out, idx_center)

    def frac_similar(self, i: int, j: int) -> float:
        """
        Fraction of neighbouring agents that are of the same type. Right now
        this only computes for one abstract class, `a`. WE CAN DO pnorm similarity
        for multiple attributes!
        """
        a = self.agent_grid[i, j]
        if a == 0: 
            return 0.0
        neigh = self.neighbours_moore(self.agent_grid, i, j)
        neigh = neigh[neigh != 0]
        if neigh.size == 0:
            return 0.0
        return float(np.count_nonzero(neigh == a) / neigh.size)

    def affordability_penalty(self, i: int, j: int) -> float:
        """
        Computes affordability penalty.
        """
        price = self.price_grid[i, j]
        income = self.income_grid[i, j]
        if self.agent_grid[i, j] == 0:
            return 0.0
        return abs(price - income) / self.scale

    def utility(self, i: int, j: int) -> float:
        """
        Computes utility, currently given by:
        U=lambda_sim * frac_similar - lambda_aff affordability_penalty
        """
        if self.agent_grid[i, j] == 0:
            return 0.0
        sim = self.frac_similar(i, j)
        pen = self.affordability_penalty(i, j)
        return self.p.similarity_weight * sim - self.p.affordability_weight * pen

    def step(self) -> int:
        H, W = self.agent_grid.shape
        empties = list(zip(*np.where(self.agent_grid == 0)))
        coords = list(zip(*np.where(self.agent_grid != 0)))
        self.rng.shuffle(coords)
        moves = 0

        # precompute utility threshold
        tau = self.p.utility_threshold

        for i, j in coords:
            u = self.utility(i, j)
            if u < tau and len(empties) > 0:
                # Move to a random empty cell
                k = self.rng.integers(len(empties))
                ei, ej = empties.pop(k)

                # Transfer agent and its income to the new cell
                self.agent_grid[ei, ej] = self.agent_grid[i, j]
                self.income_grid[ei, ej] = self.income_grid[i, j]

                # Vacate old
                self.agent_grid[i, j] = 0
                self.income_grid[i, j] = 0.0

                # Old cell becomes a new empty
                empties.append((i, j))
                moves += 1

        self.moves_per_iter.append(moves)
        # store utility snapshot (optional, can be large)
        U = np.zeros_like(self.price_grid, dtype=float)
        for i in range(H):
            for j in range(W):
                if self.agent_grid[i, j] != 0:
                    U[i, j] = self.utility(i, j)
        self.history_utility.append(U)
        self.history_agents.append(self.agent_grid.copy())
        return moves

    def run(self) -> int:
        for it in range(self.p.n_iterations):
            m = self.step()
            if m == 0:
                return it + 1
        return self.p.n_iterations

    def summary(self) -> dict:
        H, W = self.agent_grid.shape
        sims = []
        pens = []
        utils = []
        for i in range(H):
            for j in range(W):
                if self.agent_grid[i, j] == 0:
                    continue
                sims.append(self.frac_similar(i, j))
                pen = self.affordability_penalty(i, j)
                pens.append(pen)
                utils.append(self.utility(i, j))
        return {
            "avg_similarity": float(np.mean(sims)) if sims else 0.0,
            "avg_afford_penalty": float(np.mean(pens)) if pens else 0.0,
            "avg_utility": float(np.mean(utils)) if utils else 0.0,
            "moves_last_iter": self.moves_per_iter[-1] if self.moves_per_iter else 0,
        }

    def plot_agents(self):
        arr_init = self.history_agents[0].copy()
        arr_final = self.history_agents[-1].copy()
        cmap = ListedColormap(["#4db8ff", "#ffffff", "#f3d55c"])
        bounds = [-1.5, -0.5, 0.5, 1.5] # defines three intervals for -1,0,1
        norm = BoundaryNorm(bounds, cmap.N)
    
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # --- Initial state ---
        im0 = axes[0].imshow(arr_init, cmap=cmap, norm=norm, interpolation="nearest")
        axes[0].set_title("Initial state")
        axes[0].axis("off")

        # --- Final state ---
        im1 = axes[1].imshow(arr_final, cmap=cmap, norm=norm, interpolation="nearest")
        axes[1].set_title("Final state")
        axes[1].axis("off")

        # Legend (only once, to the right)
        legend_elements = [
            Patch(facecolor="#4db8ff", edgecolor="black", label="Cultural group -1"),
            Patch(facecolor="#ffffff", edgecolor="black", label="Unoccupied"),
            Patch(facecolor="#f3d55c", edgecolor="black", label="Cultural group +1")
        ]
        fig.legend(handles=legend_elements,
                loc="center right",
                bbox_to_anchor=(1.12, 0.5))

        plt.tight_layout()
        plt.show()