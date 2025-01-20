import pandas as pd
import numpy as np
from random import random
import pickle as pk

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV

from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score

def repeatedCrossVal(estimator, param_grid, X, y, n_splits, n_repeats, random_state=12345): 
    
    results_dict = dict()
    results_dict['cindex'] = []
    results_dict['times'] = []
    results_dict['auc'] = []
    results_dict['mean_auc'] = []
    results_dict['brier'] = []
    tuned_params = []
    predictions = []
    
    outer_cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    y_class = np.array([_y[0] for _y in y])

    for i, (train_index, test_index) in enumerate(outer_cv.split(X, y_class)):
        print(f'\tSplit {i}')

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        inner_cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=inner_cv, verbose=1)
        result = grid_search.fit(X_train, y_train)
        best_model = result.best_estimator_
        
        tuned_params.append(result)
        predictions.append(best_model.predict(X_test))
        
        #cindex
        results_dict['cindex'].append(best_model.score(X_test, y_test))
        
        #cumulative dynamic auc
        min_t = min([t for _,t in y_test])
        max_t = max([t for _,t in y_test])
        _times = np.arange(min_t, max_t, 15)
        results_dict['times'].append(_times)
        risk_scores = best_model.predict(X_test)
        _auc, _mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, _times)
        results_dict['auc'].append(_auc)
        results_dict['mean_auc'].append(_mean_auc)
        
        #brier score
        if hasattr(best_model, 'predict_survival_function'): 

            survs = best_model.predict_survival_function(X_test)

            min_surv = min([t for t in survs[0].x])
            max_surv = max([t for t in survs[0].x])

            min_t = max([min_t, min_surv])
            max_t = min([max_t, max_surv])

            _times = np.arange(min_t, max_t)
            preds = np.asarray([[fn(t) for t in _times] for fn in survs])

            brier_score = integrated_brier_score(y, y_test, preds, _times)
            results_dict['brier'].append(brier_score)
        
    return results_dict, predictions, tuned_params

if __name__=='__main__': 
    
    random_state=12345
    models = {'random_survival_forest': (RandomSurvivalForest(random_state=random_state), 
                                         {'min_samples_split': range(3, 11), 
                                          'min_samples_leaf': range(3, 11), 
                                          'max_features': ['sqrt', 'log2', None]}),
              'gradient_boosting': (GradientBoostingSurvivalAnalysis(random_state=random_state), 
                                    {'loss': ['coxph', 'squared', 'ipcwls'], 
                                     'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                                     'n_estimators': range(50, 201, 25), 
                                     'max_features': ['sqrt', 'log2', None], 
                                     'max_depth': list(range(1,15))}), #no admite None ni inf
             }
    

    
    #different combinations of preprocess/codification
    general_dataframes = [ 'ohe_norm']
    #cf_dataframes = ['ordinal_norm', 'bool', 'ohe']
    cf_dataframes = ['bool']
    
    for general_key in general_dataframes: 
        for cf_key in cf_dataframes: 
            X = pk.load(open(f'data/X_admissions_general_pp=({general_key})_cf_pp=({cf_key}).df', 'rb'))

            
            y = pk.load(open("data/y_admissions.df", "rb"))
            y = np.array([(i,t) for i,t in zip(y["Ingreso"], y["t"])], dtype=[('Ingreso','?'), ('t', '<f8')])
            
            for key_model in models: 
                model, param_grid = models[key_model]
                print(f'MODEL -> {key_model}, general_pp -> {general_key}, cf_pp -> {cf_key}')
                results_dict, predictions, tuned_params = repeatedCrossVal(model, param_grid, X, y, 
                                                                           n_splits=2, n_repeats=2, 
                                                                           random_state=random_state)
                
                with open(f'results/scores_model=({key_model})_general_pp=({general_key})_cf_pp=({cf_key}).pk', 'wb') as f: 
                    pk.dump(results_dict, f)
                with open(f'results/predictions_model=({key_model})_general_pp=({general_key})_cf_pp=({cf_key}).pk', 'wb') as f: 
                    pk.dump(predictions, f)
                with open(f'results/tuned_params_model=({key_model})_general_pp=({general_key})_cf_pp=({cf_key}).pk', 'wb') as f: 
                    pk.dump(tuned_params, f)