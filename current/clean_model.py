import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
import matplotlib.animation as animation
from matplotlib import colors
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# parameters
np.random.seed(42) 
n = 100 
m = 100 
house_range = 5
income_values = [0.1, 0.5, 1.0]
income_probabilities = [0.5, 0.3, 0.2]

amenities_names_fixed = ['shop', 'supermarket', 'post office', 'gym', 'school']


def create_edinburgh_map(n, m):
    city_map = np.full((n, m), None, dtype=object)

    I, J = np.indices((n, m))
    parks = np.zeros((n, m), dtype=bool)

    # oval that can be roatated to model the empty cells
    
    def rotated_oval(centre_i, centre_j, ri, rj, theta_deg):
        theta = np.deg2rad(theta_deg)
        Y = I - centre_i   # rows
        X = J - centre_j   # cols
        Xr = X * np.cos(theta) + Y * np.sin(theta)
        Yr = -X * np.sin(theta) + Y * np.cos(theta)
        return (Xr / rj) ** 2 + (Yr / ri) ** 2 <= 1.0

    meadows = rotated_oval(
        centre_i=82,   # move it down a little
        centre_j=53,
        ri=4,          # slightly thinner
        rj=26,         # a bit longer
        theta_deg=-10
    )

    # Bruntsfield Links: more clearly south-west of the Meadows
    links_main = rotated_oval(
        centre_i=88,   # a bit further south
        centre_j=30,   # slightly further west
        ri=5,
        rj=18,
        theta_deg=-20
    )

    # Tail joining Links up towards the western end of the Meadows
    tail = (
        (I >= 84) & (I <= 94) &
        (J >= 22) & (J <= 36) &
        (J <= -1.0 * (I - 94) + 36)
    )

    parks |= meadows | links_main | tail

    #other parks/emtpy cells
    psg_west = rotated_oval(centre_i=32, centre_j=42, ri=5,   rj=16,  theta_deg=-8)
    psg_east = rotated_oval(centre_i=34, centre_j=60, ri=4.5, rj=18,  theta_deg=2)
    castle_slope = rotated_oval(centre_i=28, centre_j=48, ri=6, rj=14, theta_deg=-10)
    calton = rotated_oval(centre_i=32, centre_j=84, ri=7, rj=12, theta_deg=0)

    holy_centre_i, holy_centre_j, holy_radius = 60, 100, 15
    holy_circle = ((I - holy_centre_i) ** 2 + (J - holy_centre_j) ** 2) <= holy_radius ** 2
    holyrood = holy_circle

    uni = rotated_oval(centre_i=54, centre_j=54, ri=4, rj=9, theta_deg=0)

    parks |= psg_west | psg_east | castle_slope | calton | holyrood | uni

    #road/emtpy cells
    queen_st   = (I >= 18) & (I <= 28) & (np.abs(J - (0.25 * I + 28)) <= 1)
    george_st  = (I >= 20) & (I <= 30) & (np.abs(J - (0.25 * I + 34)) <= 1)
    princes_st = (I >= 22) & (I <= 32) & (np.abs(J - (0.25 * I + 40)) <= 1)
    lothian    = (I >= 24) & (I <= 80) & (np.abs(J - (0.03 * I + 38)) <= 1)
    leith      = (I >= 18) & (I <= 60) & (np.abs(J - (0.45 * I + 30)) <= 1)
    lauriston  = (I >= 48) & (I <= 49) & (J >= 24) & (J <= 80)
    melville   = (I >= 78) & (I <= 79) & (J >= 18) & (J <= 90)
    brunts_rd  = (I >= 60) & (I <= 95) & (np.abs(J - (0.08 * I + 24)) <= 1)
    fountain   = (I >= 26) & (I <= 90) & (np.abs(J - (0.02 * I + 18)) <= 1)

    royal_mile = (np.abs(J - (0.6 * I - 5)) <= 1) & (I >= 32) & (I <= 60)
    royal_bend = (I >= 44) & (I <= 52) & (np.abs(J - (0.45 * I + 10)) <= 1)
    cowgate    = (np.abs(J - (0.6 * I - 10)) <= 1) & (I >= 40) & (I <= 65)
    main_between = (
        (I >= 40) & (I <= 99) &   # only in the gap band
        (J >= 79) & (J <= 80)     # 3-cell-wide vertical strip
    )

    roads = (queen_st | george_st | princes_st |
             lothian | leith | lauriston |
             melville | brunts_rd | fountain |
             royal_mile | royal_bend | cowgate| main_between)

    parks |= roads

# making smaller sreets for it to be more realistic
    for col in range(10, 92, 8):
        col_mask = (J == col) & (~parks) & (I >= 6) & (I <= 95)
        parks |= col_mask

    for row in range(6, 90, 8):
        row_mask = (I == row) & (~parks) & (J >= 8) & (J <= 95)
        parks |= row_mask


    stockbridge = rotated_oval(centre_i=14, centre_j=40, ri=4, rj=10, theta_deg=-5)
    inverleith  = rotated_oval(centre_i=10, centre_j=60, ri=4, rj=12, theta_deg=5)
    parks |= stockbridge | inverleith


    city_map[parks] = 'park'


    north_shops  = (I >= 8) & (I <= 14) & (J >= 30) & (J <= 70) & (~parks)
    north_super  = (I >= 12) & (I <= 15) & (J >= 48) & (J <= 52) & (~parks)
    north_school = (I >= 10) & (I <= 14) & (J >= 34) & (J <= 40) & (~parks)
    city_map[north_shops]  = 'shop'
    city_map[north_super]  = 'supermarket'
    city_map[north_school] = 'school'


    city_map[34:42, 60:70] = 'station'
    city_map[26:30, 60:66] = 'park'
    city_map[36:42, 44:52] = 'park'


    theta = np.deg2rad(-8.0)
    cx, cy = 25.0, 55.0
    X = J - cy
    Y = I - cx
    Xr = X * np.cos(theta) - Y * np.sin(theta)
    Yr = X * np.sin(theta) + Y * np.cos(theta)

    new_town_shops = (I >= 20) & (I <= 36) & (np.abs(Yr) <= 6) & (Xr >= -26) & (Xr <= 26)
    city_map[new_town_shops & (~parks)] = 'shop'

    school_blocks = [
        ((78, 44), (83, 52)),
        ((60, 58), (64, 64)),
        ((60, 26), (64, 32)),
        ((68, 40), (73, 48)),
        ((38, 34), (43, 42)),
    ]
    for (i1, j1), (i2, j2) in school_blocks:
        block_mask = (I >= i1) & (I <= i2) & (J >= j1) & (J <= j2)
        city_map[block_mask & (~parks)] = 'school'


    qm_gym = (I >= 50) & (I <= 54) & (J >= 48) & (J <= 54)
    city_map[qm_gym & (~parks)] = 'gym'

    marchmont_shops    = (I >= 84) & (I <= 90) & (J >= 44) & (J <= 62)
    bruntsfield_shops  = (I >= 82) & (I <= 90) & (J >= 14) & (J <= 30)
    city_map[marchmont_shops & (~parks)]   = 'shop'
    city_map[bruntsfield_shops & (~parks)] = 'shop'


    brunts_super    = (I >= 52) & (I <= 55) & (J >= 30) & (J <= 34)
    newington_super = (I >= 84) & (I <= 88) & (J >= 54) & (J <= 58)
    new_town_super  = (I >= 23) & (I <= 27) & (J >= 54) & (J <= 58)
    city_map[brunts_super & (~parks)]    = 'supermarket'
    city_map[newington_super & (~parks)] = 'supermarket'
    city_map[new_town_super & (~parks)]  = 'supermarket'

    # scattered amenities
    amenity_positions = [
        (48, 46, 'supermarket'), 
        (52, 32, 'supermarket'),  
        (84, 55, 'supermarket'),  
        (62, 80, 'supermarket'), 
        (24, 56, 'supermarket'),  
        (26, 64, 'supermarket'),  

        (46, 42, 'shop'),
        (60, 38, 'shop'), 
        (60, 28, 'shop'), 
        (52, 52, 'shop'),  
        (82, 44, 'shop'), 
        (30, 50, 'shop'), 
        (34, 60, 'shop'),   

        (52, 40, 'post office'),  
        (80, 56, 'post office'),
        (60, 36, 'post office'),  

        (48, 50, 'gym'), 
        (58, 62, 'gym'), 
        (54, 44, 'gym'),  

        # extra New Town amenities
        (24, 55, 'supermarket'),  
        (26, 62, 'supermarket'),   
        (25, 50, 'post office'),   
        (23, 47, 'gym'),
    ]
    for i, j, name in amenity_positions:
        if 0 <= i < n and 0 <= j < m:
            city_map[i, j] = name

    return city_map


# build map
the_map = create_edinburgh_map(n, m)
print("Amenity / city map created with shape:", the_map.shape)

percentiles = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
values = np.array([ 150000, 190000, 215000, 230000, 270000, 300000, 325000, 360000, 450000])

z = norm.ppf(percentiles)  # calculate corresponding Z-scores for percentiles
y = np.log(values) # Take log of values

z = z.reshape(-1, 1)  # make it correct shape for sklearn
model = LinearRegression(fit_intercept=True)
model.fit(z, y) # y = mu + sigma * z

mu = model.intercept_
sigma = model.coef_[0]

print("Fitted parameters:", mu, sigma)

n_agents = 10000 # number of grid spaces
samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_agents) # Create sample distribution based on log regression model

# Cap extreme outliers to avoid skew
cap = np.percentile(samples, 99)  # 99th percentile
samples_capped = np.minimum(samples, cap)

# min-max normalisation so all values are between 0 and 1
samples_normalised = (samples_capped - samples_capped.min()) / (samples_capped.max() - samples_capped.min())


# Reshape into 100 × 100 grid
value = samples_normalised.reshape(100, 100)

percentiles1 = np.array([0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80])
values1 = np.array([25684, 29501, 31521, 34014, 37976, 43169, 49409, 57746, 63386, 69876])

z1 = norm.ppf(percentiles1)  # calculate corresponding Z-scores for percentiles
y1 = np.log(values1) # Take log of values

z1 = z1.reshape(-1, 1)  # make it 2D for sklearn
model1 = LinearRegression(fit_intercept=True)
model1.fit(z1, y1) # y = mu + sigma * z

mu1 = model1.intercept_
sigma1 = model1.coef_[0]

print("Fitted parameters:", mu1, sigma1)

n_agents1 = 10000 # number of grid spaces
samples1 = np.random.lognormal(mean=mu1, sigma=sigma1, size=n_agents1)  # Create sample distribution based on log regression model

# Cap extreme outliers to avoid skew
cap1 = np.percentile(samples1, 99)  # 99th percentile
samples_capped1 = np.minimum(samples1, cap1)

# min-max normalization so all values are between 0 and 1
samples_normalised1 = (samples_capped1 - samples_capped1.min()) / (samples_capped1.max() - samples_capped1.min())

# Reshape into 100 × 100 grid
household = samples_normalised1.reshape(100, 100)


#adding in the SIMD data from other notebook
#A_amen_simd   = np.load("A_amenities.npy")
#A_score_simd  = np.load("A_matrix.npy")        
income_rank   = np.load("income_rank_matrix.npy")

#print("A_amen_simd:", A_amen_simd.shape)
#print("A_score_simd:", A_score_simd.shape)
#print("income_rank:", income_rank.shape)

# making the SIMD data income rank to the poor/middle/rich used here
#0.1 poor
#0.5 middle
# 1 rich

# Normalise ranks
r_min = income_rank.min()
r_max = income_rank.max()
r_norm = (income_rank - r_min) / (r_max - r_min + 1e-9)

# Split into terciles
low_thresh  = 1/3
high_thresh = 2/3

A_incomes_real = np.zeros_like(r_norm)

A_incomes_real[r_norm <= low_thresh]  = 0.1  # more deprived
A_incomes_real[(r_norm > low_thresh) & (r_norm <= high_thresh)] = 0.5
A_incomes_real[r_norm > high_thresh]  = 1.0  # least deprived

print("SIMD-based income grid:", A_incomes_real.shape,
      "unique values:", np.unique(A_incomes_real))



#def createlat_from_SIMD(n, m, A_incomes_real, V_real, the_map):
    #households = np.empty((n, m), dtype=object)
    #value      = np.zeros((n, m), dtype=float)

    #for i in range(n):
        #for j in range(m):
            #cell_type = the_map[i, j]

            # Non-residential cells: parks / roads / stations, etc.
            #if cell_type in ("park", "road", "station"):
                #households[i, j] = None
                #value[i, j] = 0.0
                #continue

            # Residential cell: income from SIMD, price from V_real
            #income_ij = float(A_incomes_real[i, j])
            #price_ij  = float(V_real[i, j])

            #households[i, j] = Household(income_ij)
            #value[i, j] = price_ij

    #return households, value




class Household():
    def __init__(self, income):
        ## Initialise
        self.income = income,
        self.amenities = {}

        ## Fill self.amenities
        # choose random religion/language/age category
        self.amenities['religion'] = [self.__choose_religion__(), None]
        self.amenities['language'] = [self.__choose_language__(), None]
        self.amenities['age_category'] = [self.__choose_age_category__(), None]

        # importance weights - can depend on the religion/language/age category chosen
        self.amenities['religion'][1] = self.__get_importance__('religion')
        self.amenities['language'][1] = self.__get_importance__('language')
        self.amenities['age_category'][1] = self.__get_importance__('age_category')

        # school importance depends on age_category
        self.amenities['school'] = self.__get_school_importance__()

        # fixed V_amenities weights (shop/supermarket/post office/gym)
        self.amenities['shop'] = self.__get_importance__('shop')
        self.amenities['supermarket'] = self.__get_importance__('supermarket')
        self.amenities['post office'] = self.__get_importance__('post office')
        self.amenities['gym'] = self.__get_importance__('gym')

    ### Initialisation methods
    def __get_importance__(self, amenity_name=None):
        if amenity_name=='shop':
            return 0.7
        elif amenity_name=='supermarket':
            return 0.9
        elif amenity_name=='post office':
            return 0.4
        elif amenity_name=='gym':
            if self.amenities['age_category'][0] in ['Single(s)', 'Young couple', 'Other']:
                return np.random.rand() * 0.5 + 0.5  # medium to high importance between 0.5 and 1.0
            elif self.amenities['age_category'][0] in ['Middle-aged couple', 'Family']:
                return np.random.rand() * 0.4 + 0.3  # low to medium importance between 0.3 and 0.7
            else:  # Elderly couple
                return np.random.rand() * 0.3  # low importance between 0.0 and 0.3

        return np.random.rand()  # catches other cases when amenity_name is not specified
    
    def __choose_religion__(self):
        religions = ['Christian', 'No religion', 'Muslim', 'Hindu', 'Sikh', 'Buddhist', 'Jewish', 'Other', 'Not stated']
        
        '''numbers = np.array([1717871+841053+291275, 1941116, 76737, 16379, 12795, 9055, 5997, 15196, 0])  # https://www.scotlandscensus.gov.uk/census-results/at-a-glance/religion/ - Accessed 17th November
        numbers[-1] = sum(numbers)/(100-7)*7
        probabilities = numbers / numbers.sum()'''
        probabilities = [0.53793995, 0.36636174, 0.01448316, 0.00309133, 0.0024149, 0.00170902, 0.00113186, 0.00286806, 0.06999998]

        return np.random.choice(religions, p=probabilities)
    
    def __choose_language__(self):
        languages = ['English', 'Other'] 
        
        # https://www.scotlandscensus.gov.uk/census-results/at-a-glance/languages/  - Accessed 17th November
        probabilities = [0.926, 1-0.926]  # 92.6% spoke only English in their homes

        return np.random.choice(languages, p=probabilities)
    
    def __choose_age_category__(self):
        ages = ['Single(s)', 'Young couple', 'Middle-aged couple', 'Elderly couple', 'Family', 'Other']  # Family has children, all couples do not currently have children
        
        probabilities = [0.2, 0.15, 0.1, 0.15, 0.3, 0.1]  # Made up for now

        return np.random.choice(ages, p=probabilities)
    
    def __get_school_importance__(self):

        if self.amenities['age_category'][0] in ['Family']:  # has children
            return np.random.rand() * 0.1 + 0.9  # high importance between 0.9 and 1.0
        elif self.amenities['age_category'][0] in ['Young couple', 'Elderly couple']:  # may plan to have children soon or have grandchildren they babysit
            return np.random.rand() * 0.3 + 0.5  # medium importance between 0.5 and 0.8
        else:
            return 0  # no importance
        
    ### Setters and getters - GitHub Copilot was used to generate these methods
    def set_income(self, new_income):
        self.income = (new_income,)
    
    def get_income(self):
        return self.income[0]  # for some reason income is stored as a tuple
    
    def get_amenities(self):
        """Return a shallow copy of the amenities dictionary."""
        return self.amenities.copy()

    def set_amenities(self, new_amenities):
        """Replace the entire amenities dictionary."""
        self.amenities = new_amenities

    def get_amenity(self, name):
        """Return the raw stored value for an amenity (could be list or numeric)."""
        return self.amenities.get(name, None)

    def set_amenity(self, name, value):
        """Set the raw stored value for an amenity (overwrites existing entry)."""
        self.amenities[name] = value

    def get_amenity_category(self, name):
        """
        For amenities stored as [category, importance] (e.g. 'religion', 'language', 'age_category'),
        return the category. For scalar amenities (e.g. 'school' or V), return None.
        """
        val = self.amenities.get(name)
        if isinstance(val, list) and len(val) >= 1:
            return val[0]
        return None

    def get_amenity_importance(self, name):
        """
        Return the importance/weight associated with an amenity.
        For entries stored as [category, importance] returns the importance.
        For scalar amenities returns the scalar value.
        """
        val = self.amenities.get(name)
        if isinstance(val, list) and len(val) >= 2:
            return val[1]
        return val  # might be float or None

    def set_amenity_importance(self, name, importance):
        """
        Set the importance/weight for an amenity.
        If the amenity is stored as [category, importance], updates the importance.
        Otherwise replaces the amenity value with the provided importance.
        """
        val = self.amenities.get(name)
        if isinstance(val, list) and len(val) >= 2:
            self.amenities[name][1] = importance
        else:
            self.amenities[name] = importance

    def to_dict(self):
        """Return a snapshot dictionary representation of the household (income + amenities)."""
        return {"income": self.income, "amenities": self.get_amenities()}

            ### Additional methods

    def get_amenity_importances_sum(self):
        return self.get_amenity_importance('religion') + self.get_amenity_importance('language') + self.get_amenity_importance('age_category') + self.get_amenity_importance('school')
    

class PropertyValue():
    def __init__(self, price):
        self.price = price
        self.amenities = {  # amenity in neighbourhood True/False
            'shop': False,
            'supermarket': False,
            'post office': False,
            'gym': False,
            'school': False
        }

    ### Setters and getters
    def set_price(self, new_price):
        self.price = new_price

    def get_price(self):
        return self.price

    def get_amenity_bool(self, name):
        if self.amenities[name] == True:
            return 1
        else:
            return 0

    def set_amenity(self, name, boolean):
        self.amenities[name] = boolean


def gen_amenity_matrix(prices, my_map=the_map):
    '''
    Create V matrix - combining house prices and their additional value due to amenities
    '''
    # These amenities are positive for ALL households

    amenities = np.zeros((n,m), dtype=object)

    # Fill with prices and zero amenities first
    for i in range(0, n):
        for j in range(0, m):
            amenities[i, j] = PropertyValue(float(prices[i, j]))
            
    for i in range(0, n):
        for j in range(0, m):

            # if within neighbourhood of each amenity, populate amenities matrix with += the amenity weight
            for am_name in amenities_names_fixed:

                if my_map[i][j] == am_name:
                    # Modify A to have constant amenities at 'locations' - display all same colour so set value to 0.0001
                    amenities[i][j].set_price(0.0)  # amenity locations have zero price

                    for distance_x in range(-house_range, house_range+1):
                        for distance_y in range(-house_range, house_range+1):
                            if distance_x == 0 and distance_y == 0:
                                continue  # people cannot live at the site of an amenity so zero everything
                            else:
                                new_x = i+distance_x
                                new_y = j+distance_y
                                if 0 <= new_x < n and 0 <= new_y < m:  # neighbour is in city - m and n may be wrong way around here
                                    #print(amenities[new_x][new_y])
                                    amenities[new_x][new_y].set_amenity(am_name, True)  # this amenity is in the neighbourhood of each house


    #print("Amenities Matrix:\n", amenities)
    print("Amenities Matrix:\n", get_prices_matrix(amenities))
    return amenities

def get_incomes_matrix(A):
    '''Return an n×m array of household incomes.'''
    n, m = A.shape
    incomes = np.zeros((n, m), dtype=float)
    # build boolean mask where A is Household
    mask = np.vectorize(lambda x: isinstance(x, Household))(A)
    if mask.any():
        coords = np.column_stack(np.nonzero(mask))
        for (i, j) in coords:
            incomes[i, j] = A[i, j].get_income()
    return incomes

def get_prices_matrix(V):
    '''Return an n×m array of house prices.'''
    n, m = V.shape
    prices = np.zeros((n, m), dtype=float)
    # test whether V holds floats or PropertyValue
    if isinstance(V[0,0], (int, float, np.floating)):
        return V.astype(float)
    # Otherwise extract by iterating only over non-null
    coords = np.column_stack(np.nonzero(np.ones((n, m), dtype=int)))  # all coords
    for (i, j) in coords:
        try:
            prices[i, j] = float(V[i, j].get_price())
        except Exception:
            prices[i, j] = 0.0
    return prices

def populate_new_prices(old_V, new_prices):
    V = old_V.copy()
    for i in range(n):
        for j in range(m):
            V[i,j].set_price(new_prices[i,j])
    return V

def house_update(neighbourhood_size, A, V):
    # Update house values

    print("neighbourhood_size", neighbourhood_size)

    old_V = V.copy()
    z = 0.01 # coefficient value (change as necessary) # z is lambda...    

    # Define Neighbourhood area size
    neighbourhood = np.ones((neighbourhood_size,neighbourhood_size))

    amenity_locations = np.argwhere(get_prices_matrix(V) == 0.0001)
    # Computes sum of neighbourhood house values and number of neighbours using convolution
    # constant mode with cval = 0 treats anything outside of edges of matrix as zero
    # Search up convolve function to understand this part
    neighbourhood_sum = convolve(get_prices_matrix(old_V), neighbourhood, mode='constant', cval = 0)
    neighbourhood_count = convolve(np.ones((n,m)), neighbourhood, mode='constant', cval = 0)


    # Use equation given in worksheet to calculate new values of houses
    new_prices = np.round(get_incomes_matrix(A) + z * (neighbourhood_sum / neighbourhood_count) + get_prices_matrix(V))
    # keep amenity locations fixed at 0.0
    for (ai, aj) in amenity_locations:
        new_prices[ai, aj] = 0.0
    V = populate_new_prices(V, new_prices)

    #print("Sum of Neighbourhood House Values:\n", neighbourhood_sum) 
    #print("Number of Neighbours:\n", neighbourhood_count)
    #print("Updated House Value:\n", V)
    return A, V

def delta(x1, x2, y1, y2):
    '''
    x measures how much want it
    y measures how much it costs
    '''
    d = (x1 - y1)**2 + (x2 - y2)**2 - (x1 - y2)**2 - (x2 - y1)**2 
    return d

def get_proportion_of_neighbours(A, household, loc, category, amenity_name):
    '''
    A is the matrix of households
    household is the Household object at that location
    loc is the location of the house (i,j)
    category is the category we are measuring proportion of
    '''
    global house_range  # neighbourhood size
    i = loc[0]
    j = loc[1]

    count_same = 0
    count_total = 0
    ### Additional methods

    def get_amenity_importances_sum(self):
        return self.get_amenity_importance('religion') + self.get_amenity_importance('language') + self.get_amenity_importance('age_category') + self.get_amenity_importance('school')
    

class PropertyValue():
    def __init__(self, price):
        self.price = price
        self.amenities = {  # amenity in neighbourhood True/False
            'shop': False,
            'supermarket': False,
            'post office': False,
            'gym': False,
            'school': False
        }

    ### Setters and getters
    def set_price(self, new_price):
        self.price = new_price

    def get_price(self):
        return self.price

    def get_amenity_bool(self, name):
        if self.amenities[name] == True:
            return 1
        else:
            return 0

    def set_amenity(self, name, boolean):
        self.amenities[name] = boolean


def gen_amenity_matrix(prices, my_map=the_map):
    '''
    Create V matrix - combining house prices and their additional value due to amenities
    '''
    # These amenities are positive for ALL households

    amenities = np.zeros((n,m), dtype=object)

    # Fill with prices and zero amenities first
    for i in range(0, n):
        for j in range(0, m):
            amenities[i, j] = PropertyValue(float(prices[i, j]))
            
    for i in range(0, n):
        for j in range(0, m):

            # if within neighbourhood of each amenity, populate amenities matrix with += the amenity weight
            for am_name in amenities_names_fixed:

                if my_map[i][j] == am_name:
                    # Modify A to have constant amenities at 'locations' - display all same colour so set value to 0.0001
                    amenities[i][j].set_price(0.0)  # amenity locations have zero price

                    for distance_x in range(-house_range, house_range+1):
                        for distance_y in range(-house_range, house_range+1):
                            if distance_x == 0 and distance_y == 0:
                                continue  # people cannot live at the site of an amenity so zero everything
                            else:
                                new_x = i+distance_x
                                new_y = j+distance_y
                                if 0 <= new_x < n and 0 <= new_y < m:  # neighbour is in city - m and n may be wrong way around here
                                    #print(amenities[new_x][new_y])
                                    amenities[new_x][new_y].set_amenity(am_name, True)  # this amenity is in the neighbourhood of each house


    #print("Amenities Matrix:\n", amenities)
    print("Amenities Matrix:\n", get_prices_matrix(amenities))
    return amenities

def get_incomes_matrix(A):
    """
    Return an n×m array of incomes.

    If A[i,j] is a Household, use its income.
    If A[i,j] is None or a non-residential cell, put 0.0 (or np.nan).
    """
    n, m = A.shape
    incomes_matrix = np.zeros((n, m), dtype=float)

    for i in range(n):
        for j in range(m):
            h = A[i, j]
            if isinstance(h, Household):
                incomes_matrix[i, j] = h.get_income()
            else:
                incomes_matrix[i, j] = 0.0 
                # non-residential / empty cel

    return incomes_matrix


def get_prices_matrix(V):
    n = V.shape[0]
    m = V.shape[1]
    prices_matrix = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            prices_matrix[i,j] = V[i,j].get_price()
    return prices_matrix

def populate_new_prices(old_V, new_prices):
    V = old_V.copy()
    for i in range(n):
        for j in range(m):
            V[i,j].set_price(new_prices[i,j])
    return V

def house_update(neighbourhood_size, A, V):
    # Update house values

    print("neighbourhood_size", neighbourhood_size)

    old_V = V.copy()
    z = 0.01 # coefficient value (change as necessary) # z is lambda...    

    # Define Neighbourhood area size
    neighbourhood = np.ones((neighbourhood_size,neighbourhood_size))

    amenity_locations = np.argwhere(get_prices_matrix(V) == 0.0001)
    # Computes sum of neighbourhood house values and number of neighbours using convolution
    # constant mode with cval = 0 treats anything outside of edges of matrix as zero
    # Search up convolve function to understand this part
    neighbourhood_sum = convolve(get_prices_matrix(old_V), neighbourhood, mode='constant', cval = 0)
    neighbourhood_count = convolve(np.ones((n,m)), neighbourhood, mode='constant', cval = 0)


    # Use equation given in worksheet to calculate new values of houses
    new_prices = np.round(get_incomes_matrix(A) + z * (neighbourhood_sum / neighbourhood_count) + get_prices_matrix(V))
    # keep amenity locations fixed at 0.0
    for (ai, aj) in amenity_locations:
        new_prices[ai, aj] = 0.0
    V = populate_new_prices(V, new_prices)

    print("Sum of Neighbourhood House Values:\n", neighbourhood_sum) 
    print("Number of Neighbours:\n", neighbourhood_count)
    print("Updated House Value:\n", V)
    return A, V

def delta(x1, x2, y1, y2):
    d = (x1 - y1)**2 + (x2 - y2)**2 - (x1 - y2)**2 - (x2 - y1)**2 
    return d

def get_proportion_of_neighbours(A, household, loc, category, amenity_name):
    global house_range  # neighbourhood size
    i = loc[0]
    j = loc[1]

    count_same = 0
    count_total = 0

    # Check neighbours in a 3x3 grid around the house (excluding itself)
    boundary_idx = house_range // 2
    diffs = sorted([s for s in range(-boundary_idx, boundary_idx)])
    for di in diffs:
        for dj in diffs:
            ni = i + di
            nj = j + dj
            if (di == 0 and dj == 0):
                continue  # skip itself
            else:
                if 0 <= ni < n and 0 <= nj < m:
                    neighbour_household = A[ni][nj]
                    neighbour_category = neighbour_household.get_amenity_category(amenity_name)
                    if neighbour_category == category:
                        count_same += 1
                    count_total += 1

    if count_total == 0:
        return 0  # avoid division by zero; no neighbours

    proportion = count_same / count_total
    return proportion

def get_amenities_value(A, loc1, val1, household1, loc2, val2, household2):
    '''
    loc is the location of the house (i,j)
    val is the value of the fixed amenities at that location
    household is the Household object at that location
    '''
    amenities_value1 = 0
    amenities_value2 = 0

    amenity_names_fixed = ['shop', 'supermarket', 'post office', 'gym', 'school']
    for amenity_name in amenity_names_fixed:

        # Notice these will be negative if the swap is bad for that household
        how_good_a_swap_for_h1 = household1.get_amenity_importance(amenity_name)*(val2.get_amenity_bool(amenity_name) - val1.get_amenity_bool(amenity_name))
        how_good_a_swap_for_h2 = household2.get_amenity_importance(amenity_name)*(val1.get_amenity_bool(amenity_name) - val2.get_amenity_bool(amenity_name))

        amenities_value1 += how_good_a_swap_for_h1
        amenities_value2 += how_good_a_swap_for_h2

    amenity_names_personal = ['religion', 'language', 'age_category']
    for amenity_name in amenity_names_personal:

        category1 = household1.get_amenity_category(amenity_name)
        prop1isgood = get_proportion_of_neighbours(A, household1, loc1, category1, amenity_name)
        prop2isgood = get_proportion_of_neighbours(A, household2, loc2, category1, amenity_name)
        how_good_a_swap_for_h1 = household1.get_amenity_importance(amenity_name)*(prop2isgood - prop1isgood)  # positive if prop2 is better for h1

        category2 = household2.get_amenity_category(amenity_name)
        prop1isgood = get_proportion_of_neighbours(A, household1, loc1, category2, amenity_name)
        prop2isgood = get_proportion_of_neighbours(A, household2, loc2, category2, amenity_name)
        how_good_a_swap_for_h2 = household2.get_amenity_importance(amenity_name)*(prop1isgood - prop2isgood)  # positive if prop1 is better for h2

        amenities_value1 += how_good_a_swap_for_h1
        amenities_value2 += how_good_a_swap_for_h2

    overall_amenities_value = amenities_value1 + amenities_value2

    return overall_amenities_value

def move(n,m, A, V):
    
    # Propose a move

    # Flattened choice using numpy
    total_grid_cells = n * m

    # Randomly selects a number between 0 and the total number of grid cell
    flat_indices = np.random.choice(total_grid_cells, size=2, replace=False)  # replace=False to ensure distinct choices
    # Translate value back into distinct gridspace 
    # divmod does i1 = flat_indices // m, j1 = flat_indices % m
    i1, j1 = divmod(flat_indices[0], m)
    i2, j2 = divmod(flat_indices[1], m)
    # Ensure not landing on amenity location
    while the_map[i1][j1] is not None or the_map[i2][j2] is not None:
        # Randomly selects a number between 0 and the total number of grid cells
        flat_indices = np.random.choice(total_grid_cells, size=2, replace=False)  # replace=False to ensure distinct choices
        i1, j1 = divmod(flat_indices[0], m)
        i2, j2 = divmod(flat_indices[1], m)
    
    print("Random position 1:", (i1,j1))
    print("Random position 2:", (i2,j2))

    # Calculate value of Delta function given by the worksheet
    # delta for income vs house price
    A_incomes = get_incomes_matrix(A)
    V_prices = get_prices_matrix(V)
    # DEBUG: check for NaNs
    if np.isnan(A_incomes).any():
        print("NaN found in A_incomes")
    if np.isnan(V_prices).any():
        print("NaN found in V_prices")

    print("Values used in delta money:")
    print("A1, V1:", A_incomes[i1,j1], V_prices[i1,j1])
    print("A2, V2:", A_incomes[i2,j2], V_prices[i2,j2])

    d_money = (A_incomes[i1,j1] - V_prices[i1,j1])**2 + (A_incomes[i2,j2] - V_prices[i2,j2])**2 - (A_incomes[i1,j1] - V_prices[i2,j2])**2 - (A_incomes[i2,j2] - V_prices[i1,j1])**2 
    # Modify value of house to include amenities and optional amenities
    d_amenities = get_amenities_value(A, [i1, j1], V[i1,j1], A[i1,j1], [i2, j2], V[i2,j2], A[i2,j2])

    amenities_threshold = 0.02  # minimum threshold of delta amenities to allow swap
    
    # Swap inhabitants if delta value is positive
    if d_money > 0 and d_amenities > amenities_threshold:
        A[i1,j1], A[i2,j2] = A[i2,j2], A[i1,j1]
    
    print("Delta Value (money): ", d_money)
    print("Delta Value (amenities): ", d_amenities)
    #print('Updated income',A)
    #print('Updated values',V)
    return A, V


def createlat_(n,m, income_values, income_probabilities):
    # Create matrices to keep track of house prices and corresponding house owners

    # Define lattice size

    # Create lattice of house prices
    house_values = [100, 500, 1000] # Adjust accordingly
    house_probabilities = [0.5, 0.3, 0.2] # Adjust accordingly
    V_prices = np.random.choice(house_values, size = (n,m), p = house_probabilities)

    V = gen_amenity_matrix(V_prices, the_map)

    # Create lattice of house owners
    # Poor = 0.1
    # Middle = 0.5
    # Rich = 1
    A_incomes = np.random.choice(income_values, size=(n, m), p=income_probabilities)
    # create n x m array of Household objects
    A = np.zeros((n, m), dtype=object)
    for i in range(n):
        for j in range(m):
            A[i, j] = Household(float(A_incomes[i, j]))
            cell_type = the_map[i][j]

            if cell_type in amenities_names_fixed:
                # Amenity cells: no resident, show "amenity" colour in plots
                A[i, j].set_income(0.0001)


def house_exit(A, moveout_probability, moveout_attempts, the_map):
    """
    Randomly select occupied, residential houses where the household leaves.
    We mark them as empty by setting A[i,j] = None and price unchanged.
    """
    rows, columns = A.shape

    for _ in range(moveout_attempts):
        if np.random.rand() >= moveout_probability:
            continue  # attempt fails

        i = np.random.randint(0, rows)
        j = np.random.randint(0, columns)

        # must be residential and occupied
        if the_map[i, j] in ("park", "road", "station"):
            continue
        if A[i, j] is None:
            continue

        # household leaves
        A[i, j] = None

    return A


def house_enter(A, movein_probability, movein_attempts,
                the_map, income_values, income_probabilities):
    """
    Randomly select empty residential houses and move in new households
    drawn from the specified income distribution.
    """
    rows, columns = A.shape

    for _ in range(movein_attempts):
        if np.random.rand() >= movein_probability:
            continue  # attempt fails

        i = np.random.randint(0, rows)
        j = np.random.randint(0, columns)

        # must be residential and EMPTY
        if the_map[i, j] in ("park", "road", "station"):
            continue
        if A[i, j] is not None:
            continue

        # draw a new household income
        new_inc = rng.choice(income_values, p=income_probabilities)
        A[i, j] = Household(float(new_inc))

    return A
    
def house_update(neighbourhood_size, A, V):
    """
    Update house prices based on neighbour incomes.
    Parks / roads / stations remain at price 0.
    """
    incomes = get_incomes_matrix(A)
    n, m = incomes.shape

    # convolution kernel
    kernel = np.ones((neighbourhood_size, neighbourhood_size), dtype=float)

    # sum of neighbour incomes
    neighbourhood_sum = convolve(incomes, kernel, mode='constant', cval=0.0)

    # number of contributing neighbours (non-zero incomes)
    mask_nonzero = (incomes > 0).astype(float)
    neighbourhood_count = convolve(mask_nonzero, kernel, mode='constant', cval=0.0)

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        neighbourhood_mean = np.where(
            neighbourhood_count > 0,
            neighbourhood_sum / neighbourhood_count,
            0.0
        )

    # blend current prices with neighbour means
    z = 0.5  # weight of neighbourhood effect
    new_V = (1 - z) * V + z * neighbourhood_mean

    # ensure non-residential cells stay at 0.0
    for i in range(n):
        for j in range(m):
            # if there is no household AND price was 0, keep 0
            if A[i, j] is None:
                new_V[i, j] = 0.0

    return A, new_V
def _cell_price(V, i, j):
    """Helper: return numeric price at V[i,j] whether V stores floats or PropertyValue objects."""
    v = V[i, j]
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    try:
        return float(v.get_price())
    except Exception:
        return 0.0


def cell_utility(A, V, C, i, j, similarity_weight, affordability_weight, radius):
    """
    Simple utility combining affordability and similarity.

    - affordability: negative absolute difference price vs income (normalized)
    - similarity: proportion of same-group neighbours (0..1)
    """
    # must be a household
    if not isinstance(A[i, j], Household):
        return -1e9

    income = A[i, j].get_income()
    price = _cell_price(V, i, j)

    # affordability: prefer price close to income; normalise by max(income,price,1)
    denom = max(1.0, abs(income), abs(price))
    affordability = -abs(price - income) / denom

    # similarity: fraction of neighbours with the same group label
    group_label = C[i, j]
    if group_label == 0:
        similarity = 0.0
    else:
        same = 0
        total = 0
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i + di, j + dj
                if di == 0 and dj == 0:
                    continue
                if 0 <= ni < A.shape[0] and 0 <= nj < A.shape[1]:
                    if not isinstance(A[ni, nj], Household):
                        continue
                    if C[ni, nj] == 0:
                        continue
                    total += 1
                    if C[ni, nj] == group_label:
                        same += 1
        similarity = (same / total) if total > 0 else 0.0

    return similarity_weight * similarity + affordability_weight * affordability


def propose_swap_with_schelling(A, V, C,
                                similarity_weight,
                                affordability_weight,
                                radius):
    """
    Try a single random swap between two households.
    Accept if total utility increases.
    Return (A, V, C, did_swap)
    """
    n, m = A.shape

    # Try to find two valid households (residential, occupied)
    for _ in range(100):
        i1, j1 = rng.integers(0, n), rng.integers(0, m)
        i2, j2 = rng.integers(0, n), rng.integers(0, m)
        if (i1, j1) == (i2, j2):
            continue
        if not isinstance(A[i1, j1], Household):
            continue
        if not isinstance(A[i2, j2], Household):
            continue
        # avoid amenity/non-residential cells if the_map is present
        if the_map[i1, j1] in ("park", "road", "station") or the_map[i2, j2] in ("park", "road", "station"):
            continue
        break
    else:
        # couldn't find two valid households
        return A, V, C, False

    # utility before swap
    u1_before = cell_utility(A, V, C, i1, j1, similarity_weight, affordability_weight, radius)
    u2_before = cell_utility(A, V, C, i2, j2, similarity_weight, affordability_weight, radius)

    # trial swap of households and their C labels (people carry group labels with them)
    A_trial = A.copy()
    C_trial = C.copy()
    A_trial[i1, j1], A_trial[i2, j2] = A_trial[i2, j2], A_trial[i1, j1]
    C_trial[i1, j1], C_trial[i2, j2] = C_trial[i2, j2], C_trial[i1, j1]

    u1_after = cell_utility(A_trial, V, C_trial, i1, j1, similarity_weight, affordability_weight, radius)
    u2_after = cell_utility(A_trial, V, C_trial, i2, j2, similarity_weight, affordability_weight, radius)

    if (u1_after + u2_after) > (u1_before + u2_before):
        # accept swap: copy results back into A and C
        A[:] = A_trial
        C[:] = C_trial
        return A, V, C, True

    # reject
    return A, V, C, False

def segregation_entropy(groups, habitable_mask, radius=2):
    '''
    groups : (n,m) matrix of group labels
    habitable_mask : mask matrix - true where houses can exist
    radius : neighborhood radius 

    Returns
    segregation_index is normalised local entropy over habitable cells 
                      where 1 means fully segregated and 0 means fully mixed.
    '''

    # Kernel for local neighbourhood
    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=float)
    # kernel[radius, radius] = 0  # uncomment to exclude the centre cell

    # 1. Count habitable neighbours
    neigh_hab = convolve(habitable_mask.astype(float), kernel,
                         mode='constant', cval=0.0)
    neigh_hab_safe = neigh_hab.copy()
    neigh_hab_safe[neigh_hab_safe == 0] = 1.0  # avoid division by zero

    # 2. Count neighbours of each group (here just group 1; group 0 is the rest)
    g1 = ((groups == 1) & habitable_mask).astype(float)
    neigh_g1 = convolve(g1, kernel, mode='constant', cval=0.0)

    # Local proportions
    p1 = neigh_g1 / neigh_hab_safe
    p0 = 1.0 - p1

    # 3. Shannon entropy per cell (in bits, normalised 0..1)
    eps = 1e-12
    H = -(p0 * np.log(p0 + eps) + p1 * np.log(p1 + eps))
    H_norm = H / np.log(2.0)  # max entropy for 2 groups is log(2)

    # 4. Average over habitable cells only
    H_mean = H_norm[habitable_mask].mean()

    # Segregation index: 1 - mixedness
    entropy_value = 1.0 - H_mean
    return entropy_value

def run_full_model_fake_edinburgh(
    A_init,
    V_init,
    C_init,
    the_map,
    n_steps=200,
    swaps_per_step=5000,
    moveout_probability=0.08,
    moveout_attempts=3000,
    movein_probability=0.08,
    movein_attempts=3000,
    similarity_weight=1.0,
    affordability_weight=1.0,
    radius=1,
):
    """
    Full integrated model:
      1. house price update
      2. Schelling swaps
      3. exit + entrance
    """
    A = A_init.copy()
    V = V_init.copy()
    C = C_init.copy()
    entropy_history = []

    for t in range(n_steps):

    # 1. Exit + Entrance (update who lives where)
        A = house_exit(A, moveout_probability, moveout_attempts, the_map)
        A = house_enter(
            A,
            movein_probability,
            movein_attempts,
            the_map,
            income_values=[0.1, 0.5, 1.0],
            income_probabilities=[0.3, 0.4, 0.3],
        )

        # 2. Update house prices based on neighbourhood incomes
        neighbourhood_size = 2 * radius + 1
        A, V = house_update(neighbourhood_size, A, V)

    # 3. Schelling swaps using the updated prices
        step_swaps = 0
        for _ in range(swaps_per_step):
            A, V, C, did_swap = propose_swap_with_schelling(
                A, V, C,
                similarity_weight=similarity_weight,
                affordability_weight=affordability_weight,
                radius=radius,
            )
            if did_swap:
                step_swaps += 1

        # 4. Compute and store segregation entropy
        habitable_mask = np.array([[the_map[i, j] not in ("park", "road", "station")
                                   for j in range(m)] for i in range(n)])
        S = segregation_entropy(C, habitable_mask, radius=radius)
        entropy_history.append(S)
            
        print(f"step {t}, swaps this step:", step_swaps)
    print("Simulation finished.")
    return A, V, C, entropy_history

def produce_start_end_state(A_matrix, V_matrix, the_map, state="final"):
    # Final income matrix (only residential cells)
    A_final_income = get_incomes_matrix(A_matrix).copy()

    mask_non_res = np.isin(the_map, ["park", "road", "station"])
    A_final_income[mask_non_res] = np.nan

    plt.figure(figsize=(5,5))
    plt.imshow(A_final_income, origin="lower")
    plt.title(f"{state.capitalize()} model income (residential only)")
    plt.colorbar()
    plt.savefig(f'{state}_income.png')

    # Final prices (normalised 0..1 for colour scale)
    V_final_price = V_matrix.copy().astype(float)
    V_final_norm = (V_final_price - np.nanmin(V_final_price)) / (
        np.nanmax(V_final_price) - np.nanmin(V_final_price)
    )

    V_final_norm[mask_non_res] = np.nan

    plt.figure(figsize=(5,5))
    plt.imshow(V_final_norm, origin="lower")
    plt.title(f"{state.capitalize()} model house prices")
    plt.colorbar()
    plt.savefig(f'{state}_prices.png')

def produce_entropy_results(entropy_history):

    # Plot over time
    plt.figure(figsize=(8, 5))
    plt.plot(entropy_history, marker='o', color='red')
    plt.xlabel('Timestep')
    plt.ylabel('Segregation Entropy (0 to 1)')
    plt.title('Segregation Entropy Over Time')
    plt.grid(True)
    plt.savefig('entropy_over_time.png')

    # Save data to file
    with open('entropy_over_time.txt', 'w') as f:
        f.write('Timestep ; SegregationEntropy\n')
        for t, S in enumerate(entropy_history):
            f.write(f'{t} ; {S}\n')

        f.write('\n')
        f.write('Initial Segregation Entropy: ' + str(entropy_history[0]) + '\n')
        f.write('Final Segregation Entropy: ' + str(entropy_history[-1]) + '\n')
        f.write('\n')
        f.write('Maximum Segregation Entropy: ' + str(max(entropy_history)) + '\n')
        f.write('Minimum Segregation Entropy: ' + str(min(entropy_history)) + '\n')
        

def createlat_from_SIMD(n, m, A_incomes_real, V_real, the_map):
    """
    Create households and initial values using the SIMD-derived income grid.
    the_map is your Edinburgh city map created earlier.
    This does not change any existing functions.
    """
    households = np.empty((n, m), dtype=object)
    value      = np.zeros((n, m), dtype=float)

    for i in range(n):
        for j in range(m):
            cell_type = the_map[i, j]   # uses your existing Edinburgh map

            # Parks and roads: keep them empty with zero house value
            if cell_type in ("park", "road"):
                households[i, j] = None
                value[i, j] = 0.0
                continue

            # All other cells: place a household whose income comes from SIMD
            income_ij = float(A_incomes_real[i, j])
            households[i, j] = Household(income_ij)

            # Simple initial price = income (you can comment on this choice in the write-up)
            value[i, j] = income_ij

    return households, value

n = 100
m = 100

# Build fake Edinburgh structure
the_map = create_edinburgh_map(n, m)

#  Build FAKE SIMD-like grids
# can replace this with actual A_amen_simd, A_score_simd, income_rank code
rng = np.random.default_rng(2025)

income_rank = rng.uniform(0, 1, size=(n, m))  # fake ranks 0..1
r_norm = income_rank  # already in 0..1

low_thresh  = 1/3
high_thresh = 2/3

A_incomes_real = np.zeros_like(r_norm)
A_incomes_real[r_norm <= low_thresh]      #      = 0.1  # poor
A_incomes_real[(r_norm > low_thresh) &
               (r_norm <= high_thresh)]    #     = 0.5  # middle
A_incomes_real[r_norm > high_thresh]      #      = 1.0  # rich

#print("SIMD-based income grid:", A_incomes_real.shape,
      #"unique values:", np.unique(A_incomes_real))

# Fake amenity score, for initial prices
A_amen_simd = rng.uniform(0, 1, size=(n, m))
V_real = A_amen_simd * 1000.0  # say amenities -> higher prices

#  Initial households and house values from SIMD
A0, V0 = createlat_from_SIMD(n, m, A_incomes_real, V_real, the_map)

# Initial Schelling group labels (two groups, +/-1) 
C0 = np.zeros((n, m), dtype=int)
for i in range(n):
    for j in range(m):
        if the_map[i, j] not in ("park", "road", "station") and A0[i, j] is not None:
            C0[i, j] = rng.choice([-1, 1])
        else:
            C0[i, j] = 0  # non-residential / empty

A_final, V_final, C_final, entropy_history = run_full_model_fake_edinburgh(
    A0, V0, C0, the_map,
    n_steps=500,
    swaps_per_step=5000,
    moveout_probability=0.08,
    moveout_attempts=3000,
    movein_probability=0.08,
    movein_attempts=3000,
    similarity_weight=1.0,
    affordability_weight=1.0,
    radius=1,
)

produce_start_end_state(A_final, V_final, the_map, state="final")
produce_start_end_state(A0, V0, the_map, state="initial")
produce_entropy_results(entropy_history)  # plot is saved as figure and data is saved in text file

print("Code finished.")