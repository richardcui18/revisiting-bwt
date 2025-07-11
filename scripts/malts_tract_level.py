import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import malts_src as pymalts


global DATA_PATH
DATA_PATH = '../../../../processed_data/'

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

class malts_tract_level:

    def __init__(self, 
                 city:str,
                 structure:str, 
                 year:int,
                 outcome:str = 'real') -> None:
        '''
        Settings: 
            city: 
                'chi', 'nyc'
            structure: 
                'abandoned_buildings', 'rail_stations' (chi only), 'subway_stations' (nyc only), 
                'bus_stops', 'public_schools', 'libraries', 'grocery_stores', 'restaurants'
            year: 
                2012, 2017, 2022
            outcome:
                'real', 'perceived'
        '''

        # Data 
        self._city = city 
        self._year = year 
        self._malts_df = pd.read_csv(DATA_PATH + f'malts_df_{self._year}.csv') # All required data for MALTS estimator
        self._malts_df.dropna(ignore_index = True, inplace = True) # 476 Chicago tracts lack perceived crime data
        self._malts_cols = None # Demographic features, treatment indicator, and outcome columns
        self._structure = structure + '_count'
        self._malts_df.drop(STRUCTURES.difference(set([self._structure])), axis = 1, inplace = True, errors = 'ignore')
        if outcome == 'real': 
            self._outcome = 'crime_rate'
        else: 
            self._outcome = 'perceived_crime'

        # Treatment
        self._default_treatment_threshold = self._malts_df[self._structure].median() # Set default treatment threshold as median (50-50)
        self._treatment_threshold = None # Treatment threshold, either default or manually set
        self._quantile_cut_proportion = None # Ratio to cut left and right of median

        # Match groups
        self.malts_mg_matrix = None # Match group matrix from cross validation 
        self.match_threshold = None # Number of matches in cross validation to consider two units matched
        self.m_opt = None # Distance metric weights from cross validation 

        # Results
        self.cate_ests = None # Dataframe for CATE estimates

    def set_treatment(self, threshold:int = None, quantile_cut:float = None) -> None: 
        '''
        Label tracts as treated/untreated by structure count threshold
            - Option to cut middle quantiles included for dense features: abandoned_buildings, restaurants, grocery_stores
        '''

        if not threshold:
            threshold = max(1, self._default_treatment_threshold)

        self._treatment_threshold = threshold
        self._malts_df['treatment'] = (self._malts_df[self._structure] >= self._treatment_threshold).astype(int)
        if quantile_cut is not None and self._structure in ['abandoned_buildings_count', 'restaurants_count', 'grocery_stores_count']:
            self._quantile_cut_proportion = quantile_cut
            lower_quantile = self._malts_df[self._structure].quantile(0.5 - self._quantile_cut_proportion, interpolation = 'lower')
            upper_quantile = self._malts_df[self._structure].quantile(0.5 + self._quantile_cut_proportion, interpolation = 'higher')
            self._malts_df = self._malts_df.loc[(self._malts_df[self._structure] >= lower_quantile) | (self._malts_df[self._structure] <= upper_quantile)]
        self._malts_df.reset_index(drop = True, inplace = True)
        self._malts_cols = COVARIATES + ['treatment', self._outcome]

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
        
        malts_skf = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats, random_state = 42)
        gen_skf = malts_skf.split(X = self._malts_df[self._malts_cols], y = self._malts_df['treatment'])

        mg_list = []
        self.m_opt = np.zeros(len(COVARIATES))
        self.malts_mg_matrix = pd.DataFrame(np.zeros((self._malts_df.shape[0], self._malts_df.shape[0])), 
                                     columns = self._malts_df.index, 
                                     index = self._malts_df.index)
        
        for est_idx, train_idx in gen_skf:
            df_train = self._malts_df.iloc[train_idx][self._malts_cols]
            df_est = self._malts_df.iloc[est_idx][self._malts_cols]
            m = pymalts.malts(outcome = self._outcome, 
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
            mg_i = mg_list[i]
            for a in mg_i.index:
                if a[1] != -1:
                    self.malts_mg_matrix.at[a[0], a[1]] = self.malts_mg_matrix.at[a[0], a[1]] + 1
            
        self.match_threshold = max(1, match_threshold)
        self.m_opt /= (n_splits*n_repeats)

    def _mg_diameter(self, unit, mg) -> float: 
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

        df = self._malts_df.merge(self.cate_ests[['tract_id', 'CATE']], on = 'tract_id')
        
        lower_alpha, upper_alpha = 0.025, 0.975
        lower_model = GradientBoostingRegressor(loss = 'quantile',alpha = lower_alpha)
        upper_model = GradientBoostingRegressor(loss = "quantile",alpha = upper_alpha)
        lower_model.fit(df[COVARIATES], df['CATE'])
        upper_model.fit(df[COVARIATES], df['CATE'])

        res = pd.DataFrame()
        res['tract_id'] = self._malts_df['tract_id']
        res['var_malts'] = np.abs(
            upper_model.predict(df[COVARIATES]) - lower_model.predict(df[COVARIATES])
        ) / 4
        res['var_malts'] = res['var_malts']**2

        return res
    
    def CATE(self, model:str = 'mean') -> None: 
        '''
        Calculate CATE for all treated tracts
        '''

        res_cols = ['tract_id', 'mg', 'yt', 'yc', 'CATE', 'diameter']
        self.cate_ests = pd.DataFrame(
            np.zeros((self._malts_df.shape[0], len(res_cols))), 
            columns = res_cols, 
            index = self._malts_df.index)
        self.cate_ests['mg'] = self.cate_ests['mg'].astype(object)

        thresholded_mg_matrix = self.malts_mg_matrix.where(self.malts_mg_matrix >= self.match_threshold, other = 0)
        for idx in thresholded_mg_matrix.index: 
        
            t = self._malts_df.loc[[idx]]
            x = t[COVARIATES]
            xid = t.at[idx, 'tract_id']
            self.cate_ests.at[idx, 'tract_id'] = xid
            
            # Match group
            matched_idxs = np.nonzero(thresholded_mg_matrix.loc[idx])
            mg = self._malts_df.loc[matched_idxs]
            self.cate_ests.at[idx, 'mg'] = mg['tract_id'].tolist()
            mgT, mgC = mg[mg['treatment'] == 1], mg[mg['treatment'] == 0]
            mgT_X, mgC_X = mgT[COVARIATES], mgC[COVARIATES]
            mgT_Y, mgC_Y = mgT[self._outcome], mgC[self._outcome]
            diameter = self._mg_diameter(x, mg[COVARIATES])
            self.cate_ests.at[idx, 'diameter'] = diameter

            if model == 'mean': 
                yt = np.mean(mgT_Y)
                yc = np.mean(mgC_Y)
                self.cate_ests.at[idx, 'yt'] = yt
                self.cate_ests.at[idx, 'yc'] = yc
                self.cate_ests.at[idx, 'CATE'] = yt - yc
            
            if model == 'linear': 
                yt = Ridge().fit(X = mgT_X, y = mgT_Y)
                yc = Ridge().fit(X = mgC_X, y = mgC_Y)
                yt_pred = yt.predict(x)[0]
                yc_pred = yc.predict(x)[0]
                self.cate_ests.at[idx, 'yt'] = yt_pred
                self.cate_ests.at[idx, 'yc'] = yc_pred
                self.cate_ests.at[idx, 'CATE'] = yt_pred - yc_pred

            if model == 'rf': 
                yt = RandomForestRegressor().fit(X = mgT_X, y = mgT_Y)
                yc = RandomForestRegressor().fit(X = mgC_X, y = mgC_Y)
                yt_pred = yt.predict(t[COVARIATES])[0]
                yc_pred = yc.predict(t[COVARIATES])[0]
                self.cate_ests.at[idx, 'yt'] = yt_pred
                self.cate_ests.at[idx, 'yc'] = yc_pred
                self.cate_ests.at[idx, 'CATE'] = yt_pred - yc_pred

        self.cate_ests = self.cate_ests.merge(self._malts_variance(), on = 'tract_id')
    
    def save_tables(self, path = '') -> None: 
        '''
        Save full dataframes: 
            tract_id, covariates, structure count, treatment label, outcome, match group, match group diameter, treatment/control outcome estimates, CATE, CATE variance
        '''

        fpath = path + f'{self._city}_{self._structure[:-6]}_{self._year}_{self._outcome}'
        pd.concat(
            [self.cate_ests.merge(self._malts_df, on = 'tract_id', how = 'left'), 
             self.malts_mg_matrix.rename(dict(zip(self.malts_mg_matrix.columns, self._malts_df['tract_id'])), axis = 1)],
            axis = 1
        ).to_csv(f'{fpath}.csv', index = False)
        pd.DataFrame(
            data = {
                'covariate':COVARIATES,
                'weight': self.m_opt}
        ).to_csv(f'{fpath}_mopt.csv')