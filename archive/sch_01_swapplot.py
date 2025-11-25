import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import convolve

# ==========================================
# 0. REPRODUCIBILITY SETUP
# ==========================================
np.random.seed(42) 

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
N, M = 100, 100
HOUSE_RANGE = 5
NUM_STEPS = 100          # Increased steps to see convergence better
SWAPS_PER_STEP = 5000    
LAMBDA_PRICE = 0.01      

INCOME_VALUES = np.array([0.1, 0.5, 1.0])
INCOME_PROBS = [0.5, 0.3, 0.2]

# Map Labels
EMPTY = 0
PARK = 1
RESIDENTIAL = 2
AMENITY = 3 

# ==========================================
# 2. MAP GENERATION (Vectorized)
# ==========================================
def create_fast_edinburgh_map(n, m):
    city_mask = np.full((n, m), RESIDENTIAL, dtype=np.int8)
    I, J = np.indices((n, m))
    
    def rotated_oval(ci, cj, ri, rj, theta_deg):
        theta = np.deg2rad(theta_deg)
        Y, X = I - ci, J - cj
        Xr = X * np.cos(theta) + Y * np.sin(theta)
        Yr = -X * np.sin(theta) + Y * np.cos(theta)
        return (Xr / rj) ** 2 + (Yr / ri) ** 2 <= 1.0

    parks = np.zeros((n, m), dtype=bool)
    parks |= rotated_oval(82, 53, 4, 26, -10) 
    parks |= rotated_oval(88, 30, 5, 18, -20) 
    parks |= rotated_oval(32, 42, 5, 16, -8)
    parks |= rotated_oval(34, 60, 4.5, 18, 2)
    parks |= rotated_oval(28, 48, 6, 14, -10)
    parks |= rotated_oval(32, 84, 7, 12, 0) 
    parks |= ((I - 60)**2 + (J - 100)**2 <= 15**2) 
    parks |= rotated_oval(54, 54, 4, 9, 0)
    parks |= rotated_oval(14, 40, 4, 10, -5)
    parks |= rotated_oval(10, 60, 4, 12, 5)

    roads = np.zeros((n, m), dtype=bool)
    roads |= (I >= 18) & (I <= 28) & (np.abs(J - (0.25 * I + 28)) <= 1)
    roads |= (I >= 24) & (I <= 80) & (np.abs(J - (0.03 * I + 38)) <= 1)
    roads |= (I >= 26) & (I <= 90) & (np.abs(J - (0.02 * I + 18)) <= 1)
    roads |= (np.abs(J - (0.6 * I - 5)) <= 1) & (I >= 32) & (I <= 60) 
    roads |= (J % 8 == 2) & (I >= 6) & (I <= 95)
    roads |= (I % 8 == 6) & (J >= 8) & (J <= 95)
    
    non_buildable = parks | roads
    city_mask[non_buildable] = PARK

    amenity_mask = np.zeros((n, m), dtype=bool)
    amenity_mask[(I>=8)&(I<=14)&(J>=30)&(J<=70)] = True 
    amenity_mask[(I>=50)&(I<=54)&(J>=48)&(J<=54)] = True 
    amenity_mask[(I>=82)&(I<=90)&(J>=14)&(J<=30)] = True 
    points = [(48,46), (52,32), (84,55), (62,80), (24,56), (46,42), (52,40)]
    for r, c in points:
        if 0<=r<n and 0<=c<m: amenity_mask[r, c] = True
            
    city_mask[amenity_mask & (~non_buildable)] = AMENITY
    return city_mask

# ==========================================
# 3. STATIC AMENITY SCORING
# ==========================================
def precompute_amenity_scores(city_mask, influence_range=5):
    amenity_locs = (city_mask == AMENITY)
    d = np.arange(-influence_range, influence_range + 1)
    dx, dy = np.meshgrid(d, d)
    kernel = 1.0 / (1.0 + np.sqrt(dx**2 + dy**2)) 
    kernel /= kernel.max()
    score_map = convolve(amenity_locs.astype(float), kernel, mode='constant')
    return score_map / score_map.max()

# ==========================================
# 4. INITIALIZATION
# ==========================================
def initialize_grid(city_mask):
    habitable = (city_mask == RESIDENTIAL)
    
    incomes = np.zeros((N, M))
    random_incomes = np.random.choice(INCOME_VALUES, size=(N, M), p=INCOME_PROBS)
    incomes[habitable] = random_incomes[habitable]
    
    prices = np.zeros((N, M))
    prices[habitable] = incomes[habitable] * 500 + np.random.normal(0, 50, (N,M))[habitable]
    prices[prices < 0] = 0
    
    groups = np.zeros((N, M), dtype=int)
    groups[habitable] = np.random.choice([0, 1], size=habitable.sum(), p=[0.7, 0.3])
    
    return incomes, prices, groups, habitable

# ==========================================
# 5. DYNAMICS
# ==========================================
def update_prices(prices, incomes, habitable_mask):
    kernel = np.ones((5, 5))
    kernel[2, 2] = 0 
    
    neighbor_sum = convolve(prices, kernel, mode='constant', cval=0.0)
    neighbor_count = convolve(habitable_mask.astype(float), kernel, mode='constant', cval=0.0)
    neighbor_count[neighbor_count == 0] = 1.0
    avg_neighbor_price = neighbor_sum / neighbor_count
    
    new_prices = prices + incomes + LAMBDA_PRICE * avg_neighbor_price
    final_prices = np.zeros_like(prices)
    final_prices[habitable_mask] = new_prices[habitable_mask]
    return final_prices

def get_social_score(groups, r, c, target_group):
    r_min, r_max = max(0, r-2), min(N, r+3)
    c_min, c_max = max(0, c-2), min(M, c+3)
    patch = groups[r_min:r_max, c_min:c_max]
    total = patch.size - 1 
    if total <= 0: return 0
    same = np.sum(patch == target_group)
    if groups[r,c] == target_group: same -= 1
    return same / total

def attempt_moves(incomes, prices, groups, amenity_scores, habitable_mask, n_attempts=1000):
    rows, cols = np.where(habitable_mask)
    num_houses = len(rows)
    idx1 = np.random.randint(0, num_houses, n_attempts)
    idx2 = np.random.randint(0, num_houses, n_attempts)
    
    swaps_done = 0
    
    for k in range(n_attempts):
        r1, c1 = rows[idx1[k]], cols[idx1[k]]
        r2, c2 = rows[idx2[k]], cols[idx2[k]]
        if r1 == r2 and c1 == c2: continue

        v1, v2 = prices[r1, c1], prices[r2, c2]
        a1, a2 = incomes[r1, c1], incomes[r2, c2]
        
        # Fast Delta Money
        d_money = 2 * (v1 - v2) * (a1 - a2)
        if d_money < -500: continue 

        fixed_gain_1 = amenity_scores[r2, c2] - amenity_scores[r1, c1]
        fixed_gain_2 = amenity_scores[r1, c1] - amenity_scores[r2, c2]
        
        grp1, grp2 = groups[r1, c1], groups[r2, c2]
        h1_curr = get_social_score(groups, r1, c1, grp1)
        h1_new  = get_social_score(groups, r2, c2, grp1)
        h2_curr = get_social_score(groups, r2, c2, grp2)
        h2_new  = get_social_score(groups, r1, c1, grp2)
        
        d_amenities = (h1_new - h1_curr) + (h2_new - h2_curr) + (fixed_gain_1 + fixed_gain_2) * 0.5
        
        if d_money > 0 and d_amenities > -0.1: 
            incomes[r1, c1], incomes[r2, c2] = incomes[r2, c2], incomes[r1, c1]
            groups[r1, c1], groups[r2, c2] = groups[r2, c2], groups[r1, c1]
            swaps_done += 1
            
    return incomes, groups, swaps_done

# ==========================================
# 6. MAIN SCRIPT
# ==========================================
def run_simulation():
    print("Generating Map...")
    city_map = create_fast_edinburgh_map(N, M)
    amenity_scores = precompute_amenity_scores(city_map)
    
    print("Initializing Agents...")
    incomes, prices, groups, habitable = initialize_grid(city_map)
    
    initial_incomes = incomes.copy()
    
    # Store swap history
    swap_history = []
    
    print(f"Starting Simulation ({NUM_STEPS} steps)...")
    for t in range(NUM_STEPS):
        prices = update_prices(prices, incomes, habitable)
        incomes, groups, swaps = attempt_moves(
            incomes, prices, groups, amenity_scores, habitable, n_attempts=SWAPS_PER_STEP
        )
        swap_history.append(swaps)
        
        if t % 10 == 0:
            print(f"Step {t}: Swaps = {swaps}")

    return city_map, initial_incomes, incomes, swap_history

# ==========================================
# 7. PLOTTING
# ==========================================
def plot_results(city_map, init_inc, final_inc, swap_history):
    mask_static = (city_map == PARK) | (city_map == EMPTY)
    
    # Color settings
    cmap = colors.ListedColormap(['lightgrey', '#6126b4', '#ffcc00', '#0099ff'])
    bounds = [0, 0.05, 0.2, 0.8, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # 1. SEGREGATION MAPS
    # We use constrained_layout to prevent overlap
    fig1, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    def prep_img(inc_grid):
        disp = inc_grid.copy()
        disp[mask_static] = 0.0
        return disp

    im1 = axes[0].imshow(prep_img(init_inc), cmap=cmap, norm=norm)
    axes[0].set_title("Initial Segregation (T=0)")
    axes[0].axis('off')
    
    im2 = axes[1].imshow(prep_img(final_inc), cmap=cmap, norm=norm)
    axes[1].set_title(f"Final Segregation (T={NUM_STEPS})")
    axes[1].axis('off')
    
    # LEGEND FIX: Place colorbar at the bottom horizontally
    cbar = fig1.colorbar(im1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05, aspect=40)
    cbar.set_ticks([0.12, 0.5, 1.1])
    cbar.set_ticklabels(['Poor', 'Middle', 'Rich'])
    
    plt.suptitle("Schelling Segregation on Edinburgh Topology", fontsize=16)
    
    # 2. SWAPS VS TIME PLOT
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(range(len(swap_history)), swap_history, color='#d62728', linewidth=2)
    ax2.set_title("Agent Swaps over Time")
    ax2.set_xlabel("Time Step (Price Update Iteration)")
    ax2.set_ylabel("Number of Swaps Performed")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()

if __name__ == "__main__":
    city, start_inc, end_inc, swaps = run_simulation()
    plot_results(city, start_inc, end_inc, swaps)