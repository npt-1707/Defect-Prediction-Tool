from .utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
import pickle, pandas as pd, numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def objective_func(k, train_feature, train_label, valid_feature, valid_label):
    smote = SMOTE(random_state=42, k_neighbors=int(np.round(k)))
    train_feature_res, train_label_res = smote.fit_resample(train_feature, train_label)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(train_feature_res, train_label_res)

    prob = clf.predict_proba(valid_feature)[:, 1]
    auc = roc_auc_score(valid_label, prob)

    return -auc


class JITLineModel:
    def __init__(self, language=None, load_path=None):
        self.model = RandomForestClassifier(n_estimators=300, random_state=42)
        self.count_vect = CountVectorizer(min_df=3, ngram_range=(1, 1))
        self.language = language
        if load_path is not None:
            with open(load_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved['model']
                self.count_vect = saved['count_vect']
                self.language = saved['language']
                self.label = saved['label']
        
    def fit(self, df_train_feature, train_code_change):
        '''
        df_train_feature: pd.DataFrame of 14 features of commits with columns
            ['_id', 'bug', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']
        train_code_change: list of code changed of commits
            {
                "commit id": list of commit ids,
                "code change": list of code hunk of commits 
                                (code hunk: {"added_code": list of added lines, 
                                            "removed_code": list of removed lines}})
            }
        '''
        # preprocess code change
        commit_codes = train_code_change["code change"]    
        assert len(commit_codes) > 0, "JITLine: Training commit code change is empty"
        combined_code = preprocess_code_diff(commit_codes, language=self.language)     
        self.count_vect.fit(combined_code)
        
        # concat code and features
        feature_df = df_train_feature[['_id', 'bug', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
        code_df = pd.DataFrame()
        code_df["commit_id"] = train_code_change["commit id"]
        code_df["code"] = combined_code
        code_df = code_df.sort_values(by='commit_id')
        
        feature_df = feature_df.sort_values(by='_id')
        label = feature_df['bug']
        feature_df = feature_df.drop(['_id', "bug"], axis=1)
        
        code_arr = self.count_vect.transform(code_df["code"]).astype(np.int16).toarray()
        feature_arr = feature_df.to_numpy(dtype=np.float32)
        label = label.to_numpy(dtype=np.int16)        
        final_features = np.concatenate((code_arr, feature_arr), axis=1)
        
        # split train and validation set
        X_train, X_val, y_train, y_val = train_test_split(final_features, label, test_size=0.2, random_state=42)
        
        # apply a SMOTE technique that is optimized by a Differential Evolution
        bounds = [(1, 20)]
        result = differential_evolution(objective_func, bounds, args=(X_train, y_train, X_val, y_val),
                                    popsize=10, mutation=0.7, recombination=0.3, seed=0)
        smote = SMOTE(random_state=42, k_neighbors=int(np.round(result.x)))
        
        # resample the training data
        X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
        # smote = SMOTE(random_state=42)
        # X_train_resample, y_train_resample = smote.fit_resample(final_features, label)
        self.label = y_train_resample
        self.model.fit(X_train_resample, y_train_resample)
        
    def predict_proba(self, df_test_feature, test_code_change):
        commit_codes = test_code_change["code change"]
        assert len(commit_codes) > 0, "JITLine: Testing commit code change is empty"
        combined_code = preprocess_code_diff(commit_codes, language=self.language)
        
        feature_df = df_test_feature[['_id', 'ns', 'nd', 'nf', 'entrophy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']]
        feature_df = feature_df.sort_values(by='_id')
        feature_df = feature_df.drop(['_id'], axis=1)
        code_df = pd.DataFrame()
        code_df["commit_id"] = test_code_change["commit id"]
        code_df["code"] = combined_code
        code_df = code_df.sort_values(by='commit_id')
        code_arr = self.count_vect.transform(code_df["code"]).astype(np.int16).toarray()
        feature_arr = feature_df.to_numpy(dtype=np.float32)
        
        final_features = np.concatenate((code_arr, feature_arr), axis=1)
        return self.model.predict_proba(final_features)
        
    def save(self, save_path):
        saved = {
            'model': self.model,
            'count_vect': self.count_vect,
            'language': self.language,
            'label': self.label
        }
        with open(save_path, 'wb') as f:
            pickle.dump(saved, f)
            
            
