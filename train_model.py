from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import f1_score,accuracy_score
from xgboost import XGBClassifier
from  sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import argparse
import pickle
import lightgbm as lgb
import optuna

SEED=95

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")


X,y=np.array(data["embeddings"]).reshape(-1,2048),labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

model,score=None,0

'''
def rf_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 1000, log=True)
    max_depth = trial.suggest_int("max_depth", 4, 128)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=SEED,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study_Rf = optuna.create_study(study_name="Rf_Face",direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_Rf.optimize(rf_objective, n_trials=100, show_progress_bar=True)

print("Best Trial:")
print(f" Value: {study_Rf.best_trial.value:.4f}")
print(" Params: ")
print(study_Rf.best_params)
recognizer = RandomForestClassifier(**study_Rf.best_params,random_state=SEED)'''


Rf_best_params= {'n_estimators': 645, 'max_depth': 73, 'min_samples_split': 3, 'min_samples_leaf': 1}
recognizer = RandomForestClassifier(**Rf_best_params,random_state=SEED)
recognizer.fit(X_train,y_train)
score=accuracy_score(y_test,recognizer.predict(X_test))
model=recognizer
print('Test accuracy score for Rf is '+str(accuracy_score(y_test,recognizer.predict(X_test))))



'''
def lgbm_objective(trial):
    lgbm_params = {
    "n_estimators": 2000,
    "subsample": trial.suggest_float("subsample", 0.3, 0.9),
    "min_child_samples": trial.suggest_int("min_child_samples", 60, 100),
    "max_depth": trial.suggest_int("max_depth", 4, 25),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
    "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 0.1),
    "lambda_l2": trial.suggest_float("lambda_l2", 0.001, 0.1),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0)
        }
    lgbm_model = lgb.LGBMClassifier(**lgbm_params, random_state=SEED, verbose=-1)
    lgbm_model.fit(X_train, y_train)
    y_pred = lgbm_model.predict(X_test)
    return accuracy_score(y_test, y_pred)

study_LGBM = optuna.create_study(study_name="LGBM_Face", direction="maximize")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_LGBM.optimize(lgbm_objective, n_trials=100, show_progress_bar=True)

print("Best trial:", study_LGBM.best_trial)
print("Best parameters:", study_LGBM.best_params)

recognizer = lgb.LGBMClassifier(**study_LGBM.best_params,n_estimators=2000,random_state=SEED,verbose=-1)'''

Lgbm_best_params={'subsample': 0.8362328175629586, 'min_child_samples': 96, 'max_depth': 13, 'learning_rate': 0.05562591833131212, 'lambda_l1': 0.0010836834149713813, 'lambda_l2': 0.07640755936670479, 'colsample_bytree': 0.38843126319736054}
recognizer = lgb.LGBMClassifier(**Lgbm_best_params,n_estimators=2000,random_state=SEED,verbose=-1)


recognizer.fit(X_train, y_train)
if score<accuracy_score(y_test,recognizer.predict(X_test)):
    model=recognizer
    score=accuracy_score(y_test,recognizer.predict(X_test))
print('Test accuracy score for LGBM is '+str(accuracy_score(y_test,recognizer.predict(X_test))))


def xgb_objective(trial):
    param = {
    'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3,log=True),
    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
    'max_depth': trial.suggest_int('max_depth', 3, 12),
    'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
    'gamma': trial.suggest_float('gamma', 0, 5),
    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
    'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
    }
    model = XGBClassifier(**param, random_state=SEED)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

study_XGB = optuna.create_study(study_name='XGB_Face',direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_XGB.optimize(xgb_objective, n_trials=100, show_progress_bar=True)

print("Best Trial:")
print(f" Value: {study_XGB.best_trial.value:.4f}")
print(" Params: ")
print(study_XGB.best_params)
    

recognizer = XGBClassifier(**study_XGB.best_params,random_state=SEED,verbose=-1)

'''
Xgb_best_params={}
recognizer = XGBClassifier(**Xgb_best_params,random_state=SEED,verbose=-1)'''



recognizer.fit(X_train, y_train)
if score<accuracy_score(y_test,recognizer.predict(X_test)):
    model=recognizer
    score=accuracy_score(y_test,recognizer.predict(X_test))
print('Test accuracy score for XGB is '+str(accuracy_score(y_test,recognizer.predict(X_test))))




print('Test accuracy score is '+str(score))


# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(model))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()