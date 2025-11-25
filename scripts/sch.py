import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.ndimage import convolve, binary_dilation
# ==========================================
# 0. REPRODUCIBILITY SETUP
# ==========================================
# This ensures the random initialization and random swaps 
# happen in the exact same order every time you run the script.
np.random.seed(42)

# ==========================================
# 1. SETUP & CONSTANTS
# ==========================================
N, M = 100, 100
HOUSE_RANGE = 5
#NUM_STEPS = 50           # Simulation steps (iterations of price updates)
NUM_STEPS = 5000           # Simulation steps (iterations of price updates)
SWAPS_PER_STEP = 5000    # How many agent moves to attempt per price update
LAMBDA_PRICE = 0.01      # Influence of neighborhood price

# Income classes and probabilities (Poor, Middle, Rich)
INCOME_VALUES = np.array([0.1, 0.5, 1.0])
INCOME_PROBS = [0.5, 0.3, 0.2]

# Map Labels
EMPTY = 0
PARK = 1
RESIDENTIAL = 2
AMENITY = 3  # General category for drawing

# ==========================================
# 2. MAP GENERATION (Vectorized)
# ==========================================
def create_fast_edinburgh_map(n, m):
    # 0: Empty/Road, 1: Park, 2: Residential, 3: Fixed Amenity
    city_mask = np.full((n, m), RESIDENTIAL, dtype=np.int8)
    
    I, J = np.indices((n, m))
    
    # Helper for shapes
    def rotated_oval(ci, cj, ri, rj, theta_deg):
        theta = np.deg2rad(theta_deg)
        Y, X = I - ci, J - cj
        Xr = X * np.cos(theta) + Y * np.sin(theta)
        Yr = -X * np.sin(theta) + Y * np.cos(theta)
        return (Xr / rj) ** 2 + (Yr / ri) ** 2 <= 1.0

    # --- Parks ---
    parks = np.zeros((n, m), dtype=bool)
    parks |= rotated_oval(82, 53, 4, 26, -10) # Meadows
    parks |= rotated_oval(88, 30, 5, 18, -20) # Links
    parks |= rotated_oval(32, 42, 5, 16, -8)
    parks |= rotated_oval(34, 60, 4.5, 18, 2)
    parks |= rotated_oval(28, 48, 6, 14, -10)
    parks |= rotated_oval(32, 84, 7, 12, 0) # Calton
    parks |= ((I - 60)**2 + (J - 100)**2 <= 15**2) # Holyrood
    parks |= rotated_oval(54, 54, 4, 9, 0)
    parks |= rotated_oval(14, 40, 4, 10, -5)
    parks |= rotated_oval(10, 60, 4, 12, 5)

    # Roads (Simplified vector logic)
    # Define linear roads as boolean masks
    roads = np.zeros((n, m), dtype=bool)
    roads |= (I >= 18) & (I <= 28) & (np.abs(J - (0.25 * I + 28)) <= 1)
    roads |= (I >= 24) & (I <= 80) & (np.abs(J - (0.03 * I + 38)) <= 1)
    roads |= (I >= 26) & (I <= 90) & (np.abs(J - (0.02 * I + 18)) <= 1)
    roads |= (np.abs(J - (0.6 * I - 5)) <= 1) & (I >= 32) & (I <= 60) # Royal Mile
    
    # Grid streets
    roads |= (J % 8 == 2) & (I >= 6) & (I <= 95)
    roads |= (I % 8 == 6) & (J >= 8) & (J <= 95)
    
    # Apply Parks and Roads (Non-buildable)
    non_buildable = parks | roads
    city_mask[non_buildable] = PARK

    # --- Amenities (Fixed Locations) ---
    # We treat all amenities (shop, school, gym) as "Amenity" for the mask
    # but we will calculate their value scores separately.
    amenity_mask = np.zeros((n, m), dtype=bool)
    
    # Explicit blocks
    amenity_mask[(I>=8)&(I<=14)&(J>=30)&(J<=70)] = True # North shops
    amenity_mask[(I>=50)&(I<=54)&(J>=48)&(J<=54)] = True # Gym
    amenity_mask[(I>=82)&(I<=90)&(J>=14)&(J<=30)] = True # Bruntsfield
    
    # Scattered amenities list
    points = [(48,46), (52,32), (84,55), (62,80), (24,56), (46,42), (52,40)]
    for r, c in points:
        if 0<=r<n and 0<=c<m: amenity_mask[r, c] = True
            
    city_mask[amenity_mask & (~non_buildable)] = AMENITY
    
    return city_mask

# ==========================================
# 3. STATIC AMENITY SCORING
# ==========================================
def precompute_amenity_scores(city_mask, influence_range=5):
    """
    Creates a heatmap of how 'good' a location is based on proximity to 
    fixed amenities (shops, schools, etc).
    """
    # Create a mask where amenities are located
    amenity_locs = (city_mask == AMENITY)
    
    # Create a kernel for the neighborhood
    # A generic weight matrix (higher weight if closer)
    d = np.arange(-influence_range, influence_range + 1)
    dx, dy = np.meshgrid(d, d)
    kernel = 1.0 / (1.0 + np.sqrt(dx**2 + dy**2)) # Decay function
    kernel /= kernel.max()
    
    # Convolve to get a score map
    # This represents 'how close am I to amenities'
    score_map = convolve(amenity_locs.astype(float), kernel, mode='constant')
    
    # Normalize
    score_map = score_map / score_map.max()
    return score_map

# ==========================================
# 4. INITIALIZATION
# ==========================================
def initialize_grid(city_mask):
    # Mask for habitable cells
    habitable = (city_mask == RESIDENTIAL)
    
    # 1. Incomes (A)
    incomes = np.zeros((N, M))
    random_incomes = np.random.choice(INCOME_VALUES, size=(N, M), p=INCOME_PROBS)
    incomes[habitable] = random_incomes[habitable]
    
    # 2. Prices (V)
    prices = np.zeros((N, M))
    # Initial prices roughly correlated with income + noise
    prices[habitable] = incomes[habitable] * 500 + np.random.normal(0, 50, (N,M))[habitable]
    prices[prices < 0] = 0
    
    # 3. Cultural Group (e.g. Religion/Language combined)
    # 0: Majority, 1: Minority
    groups = np.zeros((N, M), dtype=int)
    groups[habitable] = np.random.choice([0, 1], size=habitable.sum(), p=[0.7, 0.3])
    
    return incomes, prices, groups, habitable

# ==========================================
# 5. DYNAMICS (Optimized)
# ==========================================
def update_prices(prices, incomes, habitable_mask):
    """
    V(t+1) = V(t) + Income + Lambda * Avg_Neighbor_Price
    """
    kernel = np.ones((5, 5))
    kernel[2, 2] = 0 # Don't count self in neighborhood sum
    
    neighbor_sum = convolve(prices, kernel, mode='constant', cval=0.0)
    neighbor_count = convolve(habitable_mask.astype(float), kernel, mode='constant', cval=0.0)
    
    # Avoid div by zero
    neighbor_count[neighbor_count == 0] = 1.0
    avg_neighbor_price = neighbor_sum / neighbor_count
    
    # Update rule from notebook
    # Note: original code did V_new = V_old + A + ... (cumulative growth)
    new_prices = prices + incomes + LAMBDA_PRICE * avg_neighbor_price
    
    # Apply mask
    final_prices = np.zeros_like(prices)
    final_prices[habitable_mask] = new_prices[habitable_mask]
    
    return final_prices

def get_social_score(groups, r, c, target_group):
    # Fast neighborhood check using slicing
    r_min, r_max = max(0, r-2), min(N, r+3)
    c_min, c_max = max(0, c-2), min(M, c+3)
    
    patch = groups[r_min:r_max, c_min:c_max]
    total = patch.size - 1 # approximate (ignoring empty cells for speed)
    if total <= 0: return 0
    
    same = np.sum(patch == target_group)
    # Subtract self if self is the target group (approximation)
    if groups[r,c] == target_group: same -= 1
        
    return same / total

def attempt_moves(incomes, prices, groups, amenity_scores, habitable_mask, n_attempts=1000):
    """
    Performs pairwise swaps.
    Optimization: Simplified Delta Money formula.
    """
    # Get indices of all habitable houses
    rows, cols = np.where(habitable_mask)
    num_houses = len(rows)
    
    # Pick pairs of random indices
    idx1 = np.random.randint(0, num_houses, n_attempts)
    idx2 = np.random.randint(0, num_houses, n_attempts)
    
    swaps_done = 0
    
    for k in range(n_attempts):
        # Coordinates of House 1
        r1, c1 = rows[idx1[k]], cols[idx1[k]]
        # Coordinates of House 2
        r2, c2 = rows[idx2[k]], cols[idx2[k]]
        
        if r1 == r2 and c1 == c2: continue

        # 1. Delta Money
        # (A1 - V1)^2 + (A2 - V2)^2 ... simplifies to: 2(V1 - V2)(A1 - A2)
        v1, v2 = prices[r1, c1], prices[r2, c2]
        a1, a2 = incomes[r1, c1], incomes[r2, c2]
        
        d_money = 2 * (v1 - v2) * (a1 - a2)
        
        # Optimization: If money is terrible, don't bother checking neighbors
        if d_money < -500: continue 

        # 2. Delta Amenities (Social + Fixed)
        # Fixed amenity value (precomputed)
        fixed_gain_1 = amenity_scores[r2, c2] - amenity_scores[r1, c1]
        fixed_gain_2 = amenity_scores[r1, c1] - amenity_scores[r2, c2]
        
        # Social Value (Homophily)
        # Check current happiness
        grp1, grp2 = groups[r1, c1], groups[r2, c2]
        
        # H1 current vs H1 at loc 2
        h1_curr = get_social_score(groups, r1, c1, grp1)
        h1_new  = get_social_score(groups, r2, c2, grp1)
        
        # H2 current vs H2 at loc 1
        h2_curr = get_social_score(groups, r2, c2, grp2)
        h2_new  = get_social_score(groups, r1, c1, grp2)
        
        d_amenities = (h1_new - h1_curr) + (h2_new - h2_curr) + (fixed_gain_1 + fixed_gain_2) * 0.5
        
        # Threshold logic
        if d_money > 0 and d_amenities > -0.1: # Allow slight amenity drop if money is good
            # SWAP
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
    
    # Store initial state for plotting
    initial_incomes = incomes.copy()
    
    print(f"Starting Simulation ({NUM_STEPS} steps)...")
    
    for t in range(NUM_STEPS):
        # 1. Update Prices
        prices = update_prices(prices, incomes, habitable)
        
        # 2. Move Agents (Swapping)
        incomes, groups, swaps = attempt_moves(
            incomes, prices, groups, amenity_scores, habitable, n_attempts=SWAPS_PER_STEP
        )
        
        if t % 10 == 0:
            print(f"Step {t}: Swaps performed = {swaps}")

    return city_map, initial_incomes, incomes

# ==========================================
# 7. PLOTTING
# ==========================================
def plot_results(city_map, init_inc, final_inc):
    # Mask for background (Parks/Roads)
    mask_static = (city_map == PARK) | (city_map == EMPTY)
    
    # Custom colormap for incomes
    # 0.0 (Empty) -> Grey, 0.1 -> Purple, 0.5 -> Yellow, 1.0 -> Blue
    cmap = colors.ListedColormap(['lightgrey', '#6126b4', '#ffcc00', '#0099ff'])
    bounds = [0, 0.05, 0.2, 0.8, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Prepare data for display (set parks to 0 for grey color)
    def prep_img(inc_grid):
        disp = inc_grid.copy()
        disp[mask_static] = 0.0
        return disp

    # Plot Initial
    im1 = axes[0].imshow(prep_img(init_inc), cmap=cmap, norm=norm)
    axes[0].set_title("Initial Segregation (T=0)")
    axes[0].axis('off')
    
    # Plot Final
    im2 = axes[1].imshow(prep_img(final_inc), cmap=cmap, norm=norm)
    axes[1].set_title(f"Final Segregation (T={NUM_STEPS})")
    axes[1].axis('off')
    
    # Legend
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([0.12, 0.5, 1.1])
    cbar.set_ticklabels(['Poor', 'Middle', 'Rich'])
    
    plt.suptitle("Schelling Segregation on Edinburgh Topology", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    city, start_inc, end_inc = run_simulation()
    plot_results(city, start_inc, end_inc)