import sys, time, random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from lolopy.learners import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor as sk_RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

sys.path.append('.')
# import matinfotools as mit

class RandomForestRegressorCov(sk_RandomForestRegressor):
    """
    Sklearn interface which returns random forest predictions
    along with std and predictions by all estimators
    """
    def predict(self, X, return_std=True):
        preds = sk_RandomForestRegressor.predict(self, X)
        est_preds = np.empty((len(X), len(self.estimators_)))
        # loop over each tree estimator in forest and use it to predict
        for ind, est in enumerate(self.estimators_):
            est_preds[:,ind] = est.predict(X)
        if return_std:
            return preds, np.std(est_preds, axis=1)#, list(est_preds)
        else:
            return preds

        
class SLWorkflow:
    """A simulated SL workflow.
    Takes configuration dictionary as inputs.
    
    wf = SLWorkflow(sl_input_dict)
    wf.run()
    
    Get results averaged over trials like this:
    wf.df()
    Get all results like this:
    wf.results_all()
                
    """
    

    def __init__(self, sl_input_dict):
        """Set user-defined workflow settings dictionary
        keys as the workflow class attributes"""
        for k, v in sl_input_dict.items():
            setattr(self, k, v)
            
        self.data = self.dataset
        self.check_config_errors()
        
        # define the percentile range based on the inital target value. self.target_value gets upated to dist
        if type(self.target_value) in [tuple, list, np.ndarray]:
            self.percentile_range = self.target_value
        
            
    def init_percentile_range(self):
        
        
        if type(self.target_value) in [float, int]:
            # target value is the single value specified by user
            self.target_value_to_save = self.target_value
        elif type(self.target_value) in [tuple, list, np.ndarray]:
            # target value is a distribution inside specified
            # percentile range of all input materials.
#             self.percentile_range = self.target_value
            self.target_value = self.get_target_distribution()
            self.target_value_to_save = [
                self.target_value[1, 0],
                self.target_value[-2, 0]]
        else:
            raise ValueError('{} is an invalid target value.'.format(self.target_value))
            
            
        # get the best candidate which we will try to converge on
        if type(self.target_value) in [tuple, list, np.ndarray]:
            # for percentile range target values
            self.mean_percentile_val = np.mean(self.percentile_vals)
            self.best_global_candidates = self.percentile_samples
            print('{} candidates inside the {} percentile range with property {} between {} and {}'.format(
                len(self.percentile_samples),
                self.percentile_range,
                self.target_variable,
                round(np.min(self.percentile_vals), 4),
                round(np.max(self.percentile_vals), 4)))
        else:
            # for scalar target value
            best_idx = np.argmin(np.abs((
            self.data[self.target_variable] - self.target_value)))
            self.best_global_candidates = [self.data.iloc[best_idx].name]
            print('Candidate with value of {} closest to {}: {} ({})'.format(
                self.target_variable,
                self.target_value,
                self.best_global_candidates[0],
                self.data[self.target_variable].loc[self.best_global_candidates]))
        

    def get_target_distribution(self):
        """Get a target value distribution if the target value
        specified was a percentile range"""
        
        # CB - updated so target value is in middle of range
        if self.target_range:
            # get value of target in middle of range.
            quantile_target = 0.01*(self.target_range[1] - 0.5*(self.target_range[1]-self.target_range[0]))
            self.quantile_target_val = self.data[self.target_variable].quantile(quantile_target)
        
        # first sort candidates by target value
        s = self.data[self.target_variable].sort_values()
        # get indicies which define the percentile range
        idxs = [int(len(s) * p/100) for p in self.percentile_range]
        # get material candidates inside the percentile range
        self.percentile_samples = list(s.index)[idxs[0]:idxs[1]]
        # target values in the percentile range
        self.percentile_vals = list(s)[idxs[0]:idxs[1]]
        
        # get the distribution of target values
        y, x, = np.histogram(self.percentile_vals)
        # y, x, _ = plt.hist(self.percentile_vals)
        # mit.plot_setup(xlabel='Target value', ylabel='Frequency', title='Candidate values inside target percentile')
        # plt.show()
        
        # center the bins and add zeros outside the distribution
        y = np.concatenate(([0], np.ones_like(y), [0]))
        x = np.concatenate((
            [x[0] - np.diff(x)[0]/2],
            np.add(x, np.diff(x)[0]/2)[:-1],
            [x[-1] + np.diff(x)[0]/2]))
        # interpolate the target distribution so there are more points
        interp_x = np.linspace(np.min(x), np.max(x), num=100)
        interp_y = np.interp(interp_x, x, y)
        # save the distribution as a 2D array of [x, y] values
        return np.column_stack((interp_x, interp_y))
        
        

    def check_config_errors(self):
        """Check for errors in configuration of the SL workflow"""
        if self.n_iterations < 1:
            raise ValueError('{} is not a valid number of iterations.'.format(
                self.n_iterations))
        if isinstance(self.n_priors, list):
            if len(self.n_priors) < 8:
                raise ValueError('Need at least 8 n_priors for ML modeling.')
        else:
            if self.n_priors < 8:
                raise ValueError('Need at least 8 n_priors for ML modeling.')
        if self.batch_size < 1:
            raise ValueError('{} is not a valid number for batch size.'.format(
                self.batch_size))
        if self.n_trials < 1:
            raise ValueError('{} is not a valid number of trials.'.format(
                self.n_trials))
        if type(self.target_value) in [tuple, list, np.ndarray]:
            if any([self.target_value[0] >= self.target_value[1],
                    len(self.target_value) != 2]):
                raise ValueError('Target percentile range is not valid.'.format(
                    self.n_trials))   
            

    def init_sl_workflow(self):
        """Initialize the SL workflow run.
        First get the list of input variables by
        leaving out the target variable and the
        other specified variables to ignore."""
        self.start_workflow_time = time.time()
        self.input_vars = [
            v for v in list(self.data) if all([
                v != self.target_variable,
                v not in self.ignore_vars,
            ])
        ]
        print('='*50)
        return []  # return empty list to hold results
    

    def init_single_trial(self):
        """Initialize a single SL trial with the trial start time"""
        self.measured_samples = []
        self.unmeasured_candidates = list(self.data.index)
        return time.time()


    def get_unmeasured_samples(self):
        """Get a list of unmeasured chemical formulas"""
        return [i for i in list(self.data.index) if i not in self.measured_samples]
        

    def measure_samples(self, selected_candidates):
        """Measure the selected candidates"""
        self.measured_samples += selected_candidates
        self.unmeasured_samples = self.get_unmeasured_samples()
        
        
    def get_EI_scores(self, x, x0, std):
        """
        Get Expected Improvement (EI) scores given a
        target value and arrays of predictions and standard
        deviations. This methods ceates normalized (area = 1)
        Gaussian distributions for each prediction and std
        and evaluates them at the target variable.
        Inputs:
            x: value at which to compute distribution
            x0: array of centers of the distributions (predictions)
            std: array of standard deviations of distributions (uncertainties)
        """
        prefactor = 1 / np.multiply(std, np.sqrt(2 * np.pi))
        expfactor = np.exp(-np.square(np.subtract(x, x0) / std) / 2)
        return prefactor * expfactor
    
    
    def train_model(self, model, iteration):
        """Train a new model and use it for selecting new candidates"""

        X = self.data[self.input_vars].loc[self.measured_samples].values
        y = self.data[self.target_variable].loc[self.measured_samples].values
                
        model.fit(X, y)

#         feat_imp = permutation_importance(model, X, y)
#         self.random_feature_importance = feat_imp['importances_mean'][-1]
        self.random_feature_importance = model.feature_importances_[-1] #/ np.max(model.feature_importances_)
        # get fit
        #fit, fit_std = model.predict(
        #    self.data[self.input_vars].loc[self.measured_samples].values,
        #    return_std=True)
        
        # get new predictions
        new_X = self.data[self.input_vars].loc[self.unmeasured_samples].values
        new_Y = self.data[self.target_variable].loc[self.unmeasured_samples].values

        y_pred, y_std = model.predict(new_X, return_std=True)
        
        # % of compounds in predcited to be in top 10%
#         n_percent_val = self.data[self.target_variable].quantile(0.9)
#         n_top_pred = 0
#         n_top_act = 0
#         for i,j in zip(y_pred, new_Y):
#             if i >= n_percent_val and j >= n_percent_val:
#                 n_top_pred += 1
#             if j >= n_percent_val:
#                 n_top_act += 1
#         self.n_top_pred = (n_top_pred/n_top_act)

        self.std_y_all = np.std(self.data[self.target_variable].values)
        self.std_y_train = np.std(y)
        self.std_y_test = np.std(new_Y)
        
        if self.holdout_set is not None:
            holdout_X = self.holdout_set[self.input_vars].values
            holdout_Y = self.holdout_set[self.target_variable].values
            y_pred_holdout = model.predict(holdout_X, return_std=False)
#             self.gtme = mean_squared_error([np.mean(y) for i in holdout_Y], holdout_Y)**0.5
            self.gtme = np.std(holdout_Y)
            self.rmse = mean_squared_error(holdout_Y, y_pred_holdout)**0.5
        else:
            self.gtme = mean_squared_error([np.mean(y) for i in new_Y], new_Y)**0.5
            self.rmse = mean_squared_error(new_Y, y_pred)**0.5
        
        self.ndme = self.rmse/self.gtme

        self.pearsonr = pearsonr(new_Y, y_pred)
        self.spearmanr, self.spearmanr_pval = spearmanr(new_Y, y_pred)
    
        # max uncertainty samples (MU)
        uncertainties_idx = y_std.argsort()[-self.batch_size:]
        MU = [self.unmeasured_samples[s] for s in uncertainties_idx]

                
        # expected value (EV)
        if type(self.target_value) in [tuple, list, np.ndarray]:
#             pred_residuals_idx = np.abs(y_pred - self.mean_percentile_val).argsort()[:self.batch_size]
            pred_residuals_idx = np.abs(y_pred - self.quantile_target_val).argsort()[:self.batch_size]

        else:
            pred_residuals_idx = np.abs(y_pred - self.target_value).argsort()[:self.batch_size]

        EV = [self.unmeasured_samples[s] for s in pred_residuals_idx]

        # Expected Improvement (EI) 
        if type(self.target_value) in [tuple, list, np.ndarray]:
            # array of overlap curves between predicted and target distributions
            overlap = np.zeros((len(self.target_value), len(y_pred)))
            for i in range(len(y_pred)):
                overlap[:, i] = self.get_EI_scores(
                    self.target_value[:, 0],
                    y_pred[i],
                    y_std[i],
                ) * self.target_value[:, 1]
            # get integral (area) of the overlaps
            EI_unsorted = np.array([
                np.trapz(overlap[:, i],
                         x=self.target_value[:, 0]) for i in range(len(y_pred))])
        else:
            EI_unsorted = self.get_EI_scores(self.target_value, y_pred, y_std)
        EI_idx = EI_unsorted.argsort()[-self.batch_size:]
        EI = [self.unmeasured_samples[s] for s in EI_idx]

        return EV, MU, EI
    
    

    def select_candidates(self, learner, acquisition_function, iteration):
        """
        Select candidates for the current iteration.
        For the initial iteration of the SL run,
        choose samples randomly, but avoid the best sample.
        For subsequent iterations, create a model and choose 
        new candidates based on acquisition function.
        For the first iteration, we use the same random seed for all
        acquisition functions to ensure an even playing field.
        For subsequent iterations, we reset the random seed.
        """
        if iteration == 0:
            
            self.ndme = None
            self.pearsonr = None
            self.random_feature_importance = None
            self.spearmanr = None
            self.rmse = None
            self.n_top_pred = None
            self.n_targets_found = None
            self.std_y_all = None
            self.std_y_train = None
            self.std_y_test = None
            self.gtme = None

            random.seed(self.trial_random_seed)
            
            #this path was used in the diversty data testing
            if isinstance(self.n_priors, list):
                selected_candidates = [i for i in list(self.data.iloc[self.n_priors].index.to_list())]
                
            else:
                if self.poi != None:
                    # if there is a poi, select n-1 random samples + poi (i.e. force poi in training set)
                    selected_candidates = random.sample([i for i in list(self.data.index) if i not in self.best_global_candidates], self.n_priors-len(self.poi))
                    selected_candidates = selected_candidates+self.poi
                else:
                    selected_candidates = random.sample([i for i in list(self.data.index) if i not in self.best_global_candidates], self.n_priors)


        else:
            random.seed(np.random.random())           
            # train the ML model
            EV, MU, EI = self.train_model(learner, iteration)

            # choose candidates based on acquisition function
            if acquisition_function == 'Random':
                selected_candidates = random.sample(list(self.unmeasured_samples), self.batch_size)
            elif acquisition_function == 'EV':
                selected_candidates = EV
            elif acquisition_function == 'EI':
                selected_candidates = EI
            elif acquisition_function == 'MU':
                selected_candidates = MU
            else:
                raise ValueError('{} acquisition function is invalid'.format(acquisition_function))
                

        return selected_candidates

    
    def format_cols(self, df):
        """Format column titles of dataframe"""
        return [c.strip('_') for c in df.columns]

    
    def results_list_to_df(self, results_list):
        """Convert the simulated SL results to a dataframe"""
        group_by_cols = ['iteration', 'acquisition_function', 'learner']
        # columns to calculate statistics for over different trials
        agg_list = ['best_val_so_far',]# 'fraction_explored']
        # calculations to run
        agg_dict = {i: [np.mean, np.std] for i in agg_list}
        results = pd.DataFrame(results_list)
        # group by and calculate
        df = results.groupby(group_by_cols).agg(agg_dict).reset_index()
        # flatten and rename columns
        df.columns = df.columns.map('_'.join)
        df.columns = self.format_cols(df)
        return df

    
    def plot_sl_results(self, df):
        """Plot sequential learning results to compare
        performance of different acquisition functions"""
        for learner in df['learner'].unique():
            fig, ax = plt.subplots()
            for af in df['acquisition_function'].unique():
                df0 = df[(df['learner'] == learner) &
                         (df['acquisition_function'] == af)]
                signal = df0['best_val_so_far_mean']
                std = df0['best_val_so_far_std']
                plt.plot(
                    df0['iteration'],
                    signal,
                    label=af,
                    linewidth=3,
                    markersize=10,
                    marker='o')
                ax.fill_between(
                    df0['iteration'],
                    signal - std,
                    signal + std, alpha=0.2)
#             mit.plot_setup(
#                 xlabel='SL iteration',
#                 ylabel='Best value',
#                 legend=True,
#                 legend_fontsize=14,
#                 title='{}, {} trials, batch size: {}'.format(
#                     learner, self.n_trials, self.batch_size))
            fig.set_size_inches(9, 6)
           # fig.savefig('SL_plots/{}-{}-{}-{}-{}-{}-{}.png'.format(learner, self.target_variable, self.n_priors, len(self.data), 
           #                                                     self.batch_size, self.n_iterations, self.n_trials), dpi=150)
            plt.show()


    def get_best_so_far(self):
        """Get the best measured candidate and
        target value measured so far"""
        if type(self.target_value) in [tuple, list, np.ndarray]:
            # for percentile target value
#             residuals = self.data[self.target_variable].loc[self.measured_samples] - self.mean_percentile_val
            residuals = self.data[self.target_variable].loc[self.measured_samples] - self.quantile_target_val

        else:
            # for scalar targets value
            residuals = self.data[self.target_variable].loc[self.measured_samples] - self.target_value
        best_candidate_so_far = np.abs(residuals).sort_values().index[0]
        best_val_so_far = self.data[self.target_variable].loc[best_candidate_so_far]

        return best_candidate_so_far, best_val_so_far
            
            
    def run(self, verbose=True, plot_results=True, export_results=True):
        """Run the simulated SL workflow"""
        print('running...')
        results_list = self.init_sl_workflow()
        
        if type(self.target_value) in [tuple, list, np.ndarray]:
            self.target_range = self.target_value
            
        # moved to initialize at each trial so that holdout is defined before range/percentiles
#         self.init_percentile_range()
        for learner in self.learners:
            for trial in range(self.n_trials):

                self.trial_random_seed = np.random.random()
                
                # sample dataset / bootstrap
                if self.n_sample > 0:
                    self.dataset = self.dataset.sample(self.n_sample)
                    print('dataset sample size: ', self.n_sample, len(self.dataset))
                
                # sample points for holdout set
                if self.holdout_set_size != 0:
                    
                    #ensure poi does not get put in holdout set
                    if self.poi != None:
                        self.holdout_set = self.dataset.drop(self.poi).sample(self.holdout_set_size)
                    else:
                        self.holdout_set = self.dataset.sample(self.holdout_set_size)
                        
                    self.data = self.dataset.drop(self.holdout_set.index)
                else:
                    self.holdout_set = None

                self.init_percentile_range()
            
                for af in self.acquisition_functions:

                    trial_start_time = self.init_single_trial()
                    for iteration in range(self.n_iterations):
                        
                        # select new candidates
                        selected_candidates = self.select_candidates(learner, af, iteration)
                        candidate_values = list(self.data[self.target_variable].loc[selected_candidates].values)

                        # measure the selected candidates
                        self.measure_samples(selected_candidates)
                        measured_sample_values = self.data[self.target_variable].loc[self.measured_samples].values

                        # determine the best candidate measured so far
                        best_candidate_so_far, best_val_so_far = self.get_best_so_far()
                        
                        # calc number of compounds found in the target zone
                        if type(self.target_value) in [tuple, list, np.ndarray]:
                            targets_found = [x for x in measured_sample_values if x > self.percentile_vals[0] and x < self.percentile_vals[-1]]
                            
                            if iteration == 0 and len(targets_found) > 0:
                                print('target range: {}---{}'.format(self.percentile_vals[0], self.percentile_vals[-1]))
                                print('candidate values: ', candidate_values, 'targets found: ', len(targets_found))

                        else:
                            targets_found = [x for x in measured_sample_values if x == self.best_global_candidates[0]]

                        self.n_targets_found = len(targets_found)
                        
                        
                        if self.percentile_vals[0] < candidate_values[0] < self.percentile_vals[-1]:
                            self.target_found = 1
                        else:
                            self.target_found = 0
                            
                        # save results
                        results_list.append({
                            'learner': str(learner).split('()')[0],
                            'acquisition_function': af,
                            'target': self.target_variable,
                            'target_value': self.target_value_to_save,
                            'n_training': len(self.measured_samples),
                            'n_candidates': len(self.unmeasured_samples),
                            'trial': trial,
                            'iteration': iteration,
                            'best_val_so_far': best_val_so_far,
                            'best_candidate_so_far': best_candidate_so_far,
                            'workflow_runtime': time.time() - trial_start_time,
                            'selected_candidates': selected_candidates,
                            'candidate_values':candidate_values,
                            'std_y_all':self.std_y_all,
                            'std_y_train':self.std_y_train,
                            'std_y_test':self.std_y_test,
                            'RMSE':self.rmse,
                            'GTME':self.gtme,
                            'NDME':self.ndme,
                            'pearsonr':self.pearsonr,
                            'spearmanr':self.spearmanr,
                            'target_found':self.target_found,
                            'n_targets_found':self.n_targets_found,
                            'random_feature_importance':self.random_feature_importance},
                            )
                        
                        
                        # option for breaking SL cycle at discovery of first high-performer
                        if self.discovery_break_number != 0:
#                             print(af, iteration, self.n_targets_found, self.discovery_break_number)
                            if self.n_targets_found >= self.discovery_break_number:
                                break

                        # if there are not enough samples left to measure, break the SL workflow
                        if self.batch_size * 2 > len(self.unmeasured_samples):
                            break
                if verbose:
                    print('{:20}, trial {}/{} ({} min)'.format(
                        str(learner).strip('()'), trial+1, self.n_trials,
                        round((time.time() - self.start_workflow_time)/60, 2)))
        print('='*50)
        self.total_runtime = round((time.time() - self.start_workflow_time)/60, 2)
        
        print('Total time: {} min'.format(self.total_runtime))
        # save all results and trial-averaged results
        self.results_all = pd.DataFrame(results_list)
        self.df = self.results_list_to_df(results_list)
        if plot_results:
            self.plot_sl_results(self.df)