import pandas as pd 
import geopandas as gpd
import numpy as np 
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import shapely

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from custom_utils import utils
import malts_src as pymalts


global DATA_PATH
DATA_PATH = '../../../../processed_data/'

global EARTH_RADIUS
EARTH_RADIUS = 6.371e6

global STRUCTURES, COVARIATES
STRUCTURES = set(['abandoned_buildings_count', 
              'bus_stops_count', 'subway_stations_count', 'rail_stations_count',
              'libraries_count', 'public_schools_count', 
              'restaurants_count', 'grocery_stores_count'])
COVARIATES = ['total_pop', 'male_pop', 'female_pop', 'pop_density', 'num_households', 
                'white_pop', 'black_pop', 'nat_pop', 'asian_pop', 'mixed_pop', 'other_pop', 
                'no_highschool', 'highschool', 'undergrad', 'postgrad', 
                'in_labor', 'unemployed', 'median_income', 'poverty', 
                'male_never_married', 'male_married', 'male_divorced', 'female_never_married', 'female_married', 'female_divorced', 
                'm_juv', 'm_ad', 'm_eld', 'f_juv', 'f_ad', 'f_eld'] # Matched covariates

class malts_thresholder: 

    def __init__(self, 
                 city:str,
                 structure:str, 
                 year:int, 
                 radius_start:int, 
                 radius_step:float, 
                 n_radii:int, 
                 control_sample_radius:float, 
                 n_control_samples:int) -> None:
        '''
        Settings: 
            city: 
                'chi', 'nyc'
            structure: 
                'abandoned_buildings', 'rail_stations' (chi only), 'subway_stations' (nyc only), 
                'bus_stops', 'public_schools', 'libraries', 'grocery_stores', 'restaurants'
            year: 
                2012, 2017, 2022
        '''

        # Data
        self._city = city
        self._year = year 
        if city == 'chi':
            self._structure_filenames = {
                'abandoned_buildings_count': 'chicago_vacant_buildings.csv',
                'bus_stops_count': 'chicago_bus_stops.csv',
                'libraries_count': 'chicago_libraries.csv',
                'public_schools_count': 'chicago_public_schools.csv',
                'rail_stations_count': 'chicago_rail_stations.csv',
                'restaurants_count': 'chicago_restaurants.csv',
                'grocery_stores_count': 'chicago_grocery_stores.csv'
            }
        elif city == 'nyc':
            self._structure_filenames = {
                'abandoned_buildings_count': 'nyc_abandoned_buildings.csv',
                'bus_stops_count': 'nyc_bus_stops.csv',
                'libraries_count': 'nyc_libraries.csv',
                'public_schools_count': 'nyc_public_schools.csv',
                'subway_stations_count': 'nyc_subway_stations.csv',
                'restaurants_count': 'nyc_restaurant_inspections.csv',
                'grocery_stores_count': 'nyc_retail_food_stores.csv'
            }

        # Data
        self._malts_df = None # All required data for MALTS estimator
        self._malts_cols = None # Demographic features, treatment indicator, and outcome columns
        self._structure = structure + '_count'
        self._structure_df = None # Dataframe with structure date, coordinates, tract ID
        self._crime_df = None # Dataframe for individual crimes 

        # Treatment
        self._default_treatment_threshold = None # Default treatment threshold
        self._treatment_threshold = None # Treatment threshold, either default or manually set
        self._quantile_cut_proportion = None # Ratio to cut left and right of median

        # Match groups
        self.malts_mg_matrix = None # Match group matrix from cross validation 
        self.m_opt = None # Distance metric weights from cross validation 

        # Radius settings
        self._radius_start = radius_start # Min radius
        self._radius_step = radius_step # Gap between radii
        self._n_radii = n_radii # Number of radii
        self._radii = None # Array with all radii (converted to radians)

        # Control point sampling settings
        self._control_sample_radius = control_sample_radius # Radius of control sample area 
        self._n_control_samples = n_control_samples # Number of randomly sampled control points per tract

        # Results
        self.tract_level_results = None # Store thresholding results
        self.cate_ests = None # Dataframe for CATE estimates

        self._initialize_data()
        self._crime_nn = NearestNeighbors(metric = 'haversine').fit(X = self._crime_df[['lat', 'lon']]) # NearestNeighbor object for counting crimes 
        self._set_radii()

    def _initialize_data(self) -> None:
        '''
        Construct dataframes
            - malts_df: demographic data, crime rate, structure count, tract centroid
            - structure_df: structure date, coordinates, tract ID
            - crime_df: crime dates and coordinates
            - tract_df: tract geometries, IDs, centroids
        '''

        self._malts_df = pd.read_csv(DATA_PATH + f'malts_df_{self._year}.csv')
        self._malts_df.drop(STRUCTURES.difference(set([self._structure])), axis = 1, inplace = True, errors = 'ignore')
        if self._year == 2022: 
            self._malts_df.drop('perceived_crime', axis = 1, inplace = True, errors = 'ignore')

        if self._city == 'chi':
            self._crime_df = pd.read_csv(DATA_PATH + 'crime/chicago_violent_crime.csv', usecols = ['date', 'lat', 'lon'])
            self._crime_df['date'] = pd.to_datetime(self._crime_df['date'], format = '%Y')
        else:
            self._crime_df = pd.read_csv(DATA_PATH + 'crime/nypd_complaints.csv', usecols = ['date', 'lat', 'lon'])
            self._crime_df['date'] = pd.to_datetime(self._crime_df['date'], format = '%m/%d/%Y')
        self._crime_df['lat'] = utils.deg_to_rad(self._crime_df['lat'])
        self._crime_df['lon'] = utils.deg_to_rad(self._crime_df['lon'])
        # Masks for 5Y survey periods (crime is masked to exact period, feature is masked by all features existing during or before period)
        crime_mask12 = (self._crime_df['date'] >= pd.Timestamp(2008, 1, 1)) & (self._crime_df['date'] < pd.Timestamp(2013, 1, 1))
        crime_mask17 = (self._crime_df['date'] >= pd.Timestamp(2013, 1, 1)) & (self._crime_df['date'] < pd.Timestamp(2018, 1, 1))
        crime_mask22 = (self._crime_df['date'] >= pd.Timestamp(2018, 1, 1)) & (self._crime_df['date'] < pd.Timestamp(2023, 1, 1))

        self._structure_df = pd.read_csv(DATA_PATH + f'features/{self._structure_filenames[self._structure]}', 
                        usecols = ['date', 'tract_id_10', 'tract_id_20', 'lat', 'lon'])
        self._structure_df['date'] = pd.to_datetime(self._structure_df['date'])

        if self._year == 2012: 
            self._crime_df = self._crime_df[crime_mask12] # Crimes in respective time period
            self._structure_df = self._structure_df[self._structure_df['date'] < pd.Timestamp(2013, 1, 1)] # Structures in respective time period
            self._structure_df = self._structure_df.drop('tract_id_20', axis = 1).rename({'tract_id_10': 'tract_id'}, axis = 1)
            tract_df = gpd.read_file(DATA_PATH + 'census_boundaries/census_tract_boundaries_2010.json') # Tract geographic information
        
        elif self._year == 2017: 
            self._crime_df = self._crime_df[crime_mask17]
            self._structure_df = self._structure_df[self._structure_df['date'] < pd.Timestamp(2018, 1, 1)]
            self._structure_df = self._structure_df.drop('tract_id_20', axis = 1).rename({'tract_id_10': 'tract_id'}, axis = 1)
            tract_df = gpd.read_file(DATA_PATH + 'census_boundaries/census_tract_boundaries_2010.json') 
        
        else: 
            self._crime_df = self._crime_df[crime_mask22] 
            self._structure_df = self._structure_df[self._structure_df['date'] < pd.Timestamp(2023, 1, 1)] 
            self._structure_df = self._structure_df.drop('tract_id_10', axis = 1).rename({'tract_id_20': 'tract_id'}, axis = 1)
            tract_df = gpd.read_file(DATA_PATH + 'census_boundaries/census_tract_boundaries_2020.json') 
            if self._city == 'nyc':
                tract_df['tract_id'] = '1400000US' + tract_df['GEOID'].astype(str)
        
        self._structure_df['lat'] = utils.deg_to_rad(self._structure_df['lat']) # Convert geographic coordinates to radians
        self._structure_df['lon'] = utils.deg_to_rad(self._structure_df['lon'])
        self._structure_df.reset_index(drop = True, inplace = True)

        tract_df['centroid'] = tract_df['geometry'].to_crs('+proj=cea').centroid.to_crs(tract_df.crs) # Calculate centroids
        tract_df = tract_df[['tract_id', 'centroid']] 
        self._malts_df = self._malts_df.merge(tract_df, on = 'tract_id', how = 'left')
        self._malts_df.dropna(inplace = True) # Drop tracts without crime rate or centroid
        self._malts_df.reset_index(drop = True, inplace = True)

        self._default_treatment_threshold = self._malts_df[self._structure].median() # Set default treatment threshold as median (50-50)

    def set_treatment(self, threshold:int = None, quantile_cut:float = None) -> None: 
        '''
        Label tracts as treated/untreated by structure count threshold
            - Option to cut middle quantiles included for dense features: abandoned_buildings, affordable_housing, liquor_stores, restaurants, grocery_stores
        '''

        if not threshold:
            threshold = max(1, self._default_treatment_threshold)

        self._treatment_threshold = threshold
        self._malts_df['treatment'] = (self._malts_df[self._structure] >= self._treatment_threshold).astype(int)
        if quantile_cut is not None and self._structure in ['abandoned_buildings', 'affordable_housing', 'liquor_stores', 'restaurants', 'grocery_stores']:
            self._quantile_cut_proportion = quantile_cut
            lower_quantile = self._malts_df[self._structure].quantile(0.5 - self._quantile_cut_proportion, interpolation = 'lower')
            upper_quantile = self._malts_df[self._structure].quantile(0.5 + self._quantile_cut_proportion, interpolation = 'higher')
            self._malts_df = self._malts_df[(self._malts_df[self._structure] <= lower_quantile) | (self._malts_df[self._structure] >= upper_quantile)]
            self._malts_df.reset_index(drop = True, inplace = True)
        self._malts_cols = self._malts_df.columns.drop(['tract_id', self._structure, 'centroid'], errors = 'ignore') 

    def _set_radii(self) -> None:
        '''
        Convert meters to degrees of latitude/longitude and return set of evenly spaced radii
        '''

        self._radii = np.linspace(
            start = self._radius_start, 
            stop = self._radius_start + self._radius_step*(self._n_radii-1), 
            num = self._n_radii) / EARTH_RADIUS
        
    def _random_sample(self, center:tuple, radius:float) -> pd.DataFrame:
        '''
        Randomly sample n_samples points within radius of center
            - Center is (lat, lon) coordinate pair
            - Radius should be in meters
        '''

        if type(center) == shapely.geometry.point.Point:
            center = (center.y, center.x) # Coords are switched 

        center = (utils.deg_to_rad(center[0]), utils.deg_to_rad(center[1]))

        # This isn't technically uniform sampling because we're doing it over a circular region 
        r = np.random.random(size = self._n_control_samples)*radius / EARTH_RADIUS
        theta = np.random.random(size = self._n_control_samples)*2*math.pi
        origin_lat, origin_lon = center

        random_coords = pd.DataFrame()
        random_coords['lat'] = r*np.sin(theta) + origin_lat
        random_coords['lon'] = r*np.cos(theta) + origin_lon

        return random_coords
        
    def _intervaled_crime_density(self, coords:pd.DataFrame) -> pd.Series:
        '''
        Get crime density at fixed radii centered at coords 
            - coords should be a dataframe of shape (n points, 2) where each row is a (lat, lon) coordinate
        '''

        res = pd.DataFrame()

        for i, r in enumerate(self._radii):
            area = math.pi * (r*EARTH_RADIUS)**2 / 1e6
            _, idxs = self._crime_nn.radius_neighbors(X = coords[['lat', 'lon']], radius = r)
            
            crime_densities = []
            for nn_list in idxs:
                crime_densities.append(len(nn_list)/area)
            
            res[i] = crime_densities

        return res.mean()
    
    def _create_cate_df(self) -> None: 
        '''
        Create dataframe with thresholding results for CATE estimation. Returns thresholding result for each tract. Columns are:
            - Demographic data 
            - Treatment indicator
            - Average crime density at each radius (around features for treated tracts or around control points for untreated tracts)
        '''

        control_tracts = set(self._malts_df[self._malts_df['treatment'] == 0]['tract_id'])
        control_res = pd.DataFrame() # Dataframe of all control tracts and average crime density at each radius for each tract
        for tract_id in control_tracts: # Get tract_id and centroids of all control tracts
            point = self._malts_df.loc[self._malts_df['tract_id'] == tract_id]
            point = self._malts_df.loc[self._malts_df['tract_id'] == tract_id]
            sampled_points = self._random_sample(center = point['centroid'].iloc[0], radius = self._control_sample_radius) # Random sample of points in each control tract
            tract_res = self._intervaled_crime_density(sampled_points) # Thresholding treatment, averaged over all sampled points in a tract
            tract_res['tract_id'] = tract_id # Adding tract_id
            tract_res = pd.DataFrame(tract_res).transpose()
            control_res = pd.concat(objs = [control_res, tract_res], ignore_index = True)

        treated_tracts = set(self._malts_df[self._malts_df['treatment'] == 1]['tract_id'])
        treatment_res = pd.DataFrame() # Dataframe of all treated tracts and average crime density around all structures at each radius for each tract
        for tract_id in treated_tracts: 
            structure_coords = self._structure_df[self._structure_df['tract_id'] == tract_id][['lat', 'lon']] # Dataframe of coordinates of structures in a given treated tract
            tract_res = self._intervaled_crime_density(structure_coords) # Thresholding treatment, averaged over all structures in a tract
            tract_res['tract_id'] = tract_id
            tract_res = pd.DataFrame(tract_res).transpose()
            treatment_res = pd.concat(objs = [treatment_res, tract_res], ignore_index = True)

        res_df = pd.concat(objs = [control_res, treatment_res], ignore_index = True)

        self.tract_level_results = self._malts_df[['tract_id'] + list(COVARIATES) + ['treatment']].merge(res_df, on = 'tract_id')
    
    def fit_malts(self, 
                  discrete = [], 
                  C = 1, 
                  k_tr = 15, 
                  k_est = 20, 
                  reweight = False,
                  n_splits = 5,
                  n_repeats = 2,
                  match_threshold = 1) -> None:
        '''
        Fit MALTS model and get match groups with cross validation (from malts_mf class in pymalts source)
        '''
        
        # Stratified cross validation to get match groups (weighted by number of times matched) and distance metric weights (averaged over folds)
        malts_skf = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = 42)
        gen_skf = malts_skf.split(X = self._malts_df, y = self._malts_df['treatment'])

        mg_list = []
        self.m_opt = np.zeros(len(COVARIATES))
        # Match group matrix
        self.malts_mg_matrix = pd.DataFrame(np.zeros((self._malts_df.shape[0], self._malts_df.shape[0])), 
                                     columns = self._malts_df.index, 
                                     index = self._malts_df.index)
        
        for est_idx, train_idx in gen_skf:
            df_train = self._malts_df.iloc[train_idx][self._malts_cols]
            df_est = self._malts_df.iloc[est_idx][self._malts_cols]
            m = pymalts.malts(outcome = 'crime_rate', 
                              treatment = 'treatment', 
                              data = df_train, 
                              discrete = discrete, 
                              C = C, 
                              k = k_tr, 
                              reweight = reweight)
            m.fit()
            mg = m.get_matched_groups(df_est, k_est)
            mg_list.append(mg)
            self.m_opt = np.add(self.m_opt, m.M_opt.to_numpy().flatten())

        for i in range(n_splits*n_repeats):
            # Contruct match group matrix 
            mg_i = mg_list[i]
            for a in mg_i.index:
                if a[1]!=-1:
                    self.malts_mg_matrix.loc[a[0], a[1]] = self.malts_mg_matrix.loc[a[0], a[1]] + 1

        self.match_threshold = max(1, match_threshold)
        self.m_opt /= (n_splits*n_repeats) # Calculate average metric weights 

    def _mg_diameter(self, unit, mg): 
        '''
            Get diameter of match group
        '''

        s = ((mg.to_numpy() - unit.to_numpy()) * self.m_opt)**2
        dists = np.sum(s, axis = 1)
        return np.max(np.sqrt(dists))
    
    def _malts_variance(self) -> np.ndarray:
        '''
        MALTS variance estimator given in Parikh et al. (2022)
            - Requires CATE estimates
        '''

        t_cols = [f'yt_{i}' for i in range(self._radii.shape[0])]
        c_cols = [f'yc_{i}' for i in range(self._radii.shape[0])]
        df = self._malts_df[['tract_id'] + COVARIATES].merge(self.cate_ests[['tract_id'] + t_cols + c_cols], on = 'tract_id')
        
        lower_alpha, upper_alpha = 0.025, 0.975
        lower_model = GradientBoostingRegressor(loss = 'quantile',alpha = lower_alpha)
        upper_model = GradientBoostingRegressor(loss = "quantile",alpha = upper_alpha)

        res = pd.DataFrame()
        res['tract_id'] = self._malts_df['tract_id']

        for i, col in enumerate(t_cols):
            lower_model.fit(df[COVARIATES], df[col])
            upper_model.fit(df[COVARIATES], df[col])
            res[f'var_t_{i}'] = np.abs(upper_model.predict(df[COVARIATES]) - lower_model.predict(df[COVARIATES])) / 4
            res[f'var_t_{i}'] = res[f'var_t_{i}']**2

        for i, col in enumerate(c_cols):
            lower_model.fit(df[COVARIATES], df[col])
            upper_model.fit(df[COVARIATES], df[col])
            res[f'var_c_{i}'] = np.abs(upper_model.predict(df[COVARIATES]) - lower_model.predict(df[COVARIATES])) / 4
            res[f'var_c_{i}'] = res[f'var_c_{i}']**2

        return res

    def CATE(self, model:str = 'mean') -> None: 
        '''
        Calculate CATE for all treated tracts
        '''

        self._create_cate_df() 
        res_cols = ['tract_id', 'mg'] + [f'yt_{i}' for i in range(self._radii.shape[0])] + [f'yc_{i}' for i in range(self._radii.shape[0])] + [f'CATE_{i}' for i in range(self._radii.shape[0])] +['diameter']
        self.cate_ests = pd.DataFrame(np.zeros((self._malts_df.shape[0], len(res_cols))), 
                                     columns = res_cols, 
                                     index = self._malts_df.index)
        self.cate_ests['mg'] = self.cate_ests['mg'].astype(object)

        thresholded_mg_matrix = self.malts_mg_matrix.where(self.malts_mg_matrix >= self.match_threshold, other = 0)
        for idx in self.malts_mg_matrix.index: 
        
            t = self.tract_level_results.loc[[idx]]
            x = t[COVARIATES]
            xid = t.at[idx, 'tract_id']
            self.cate_ests.at[idx, 'tract_id'] = xid
            
            # Match group
            matched_idxs = np.nonzero(thresholded_mg_matrix.loc[idx])
            mg = self.tract_level_results.loc[matched_idxs]
            self.cate_ests.at[idx, 'mg'] = mg['tract_id'].tolist()
            mgT, mgC = mg[mg['treatment'] == 1], mg[mg['treatment'] == 0]
            mgT_X, mgC_X = mgT[COVARIATES], mgC[COVARIATES]
            diameter = self._mg_diameter(x, mg[COVARIATES])
            self.cate_ests.loc[idx, 'diameter'] = diameter

            if model == 'mean': 
                for i in range(self._radii.shape[0]):
                    mgT_Y, mgC_Y = mgT[i], mgC[i]
                    yt = np.mean(mgT_Y)
                    yc = np.mean(mgC_Y)
                    self.cate_ests.loc[idx, f'yt_{i}'] = yt
                    self.cate_ests.loc[idx, f'yc_{i}'] = yc
                    self.cate_ests.loc[idx, f'CATE_{i}'] = yt - yc
            
            if model == 'linear': 
                for i in range(self._radii.shape[0]):
                    mgT_Y, mgC_Y = mgT[i], mgC[i]
                    yt = Ridge().fit(X = mgT_X, y = mgT_Y)
                    yc = Ridge().fit(X = mgC_X, y = mgC_Y)
                    self.cate_ests.loc[idx, f'yt_{i}'] = yt.predict(x)[0]
                    self.cate_ests.loc[idx, f'yc_{i}'] = yc.predict(x)[0]
                    self.cate_ests.loc[idx, f'CATE_{i}'] = yt - yc

            if model == 'rf': 
                for i in range(self._radii.shape[0]):
                    mgT_Y, mgC_Y = mgT[i], mgC[i]
                    yt = RandomForestRegressor().fit(X = mgT_X, y = mgT_Y)
                    yc = RandomForestRegressor().fit(X = mgC_X, y = mgC_Y)
                    self.cate_ests.loc[idx, f'yt_{i}'] = yt.predict(x)[0]
                    self.cate_ests.loc[idx, f'yc_{i}'] = yc.predict(x)[0]
                    self.cate_ests.loc[idx, f'CATE_{i}'] = yt - yc

        self.cate_ests = self.cate_ests.merge(self._malts_variance(), on = 'tract_id')
    
    def save_tables(self, path = '') -> None: 
        '''
        Save dataframes with CATE estimates
        '''

        fpath = path + f'{self._city}_{self._structure[:-6]}_{self._year}'
        pd.concat(
            [self.cate_ests.merge(self._malts_df, on = 'tract_id', how = 'left'), 
             self.malts_mg_matrix.rename(dict(zip(self.malts_mg_matrix.columns, self._malts_df['tract_id'])), axis = 1)],
            axis = 1
        ).to_csv(f'{fpath}.csv', index = False)