from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import argparse
import pickle
import lightgbm as lgb



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

param_distributions = {
    'n_estimators': np.arange(100, 500, 50),
    'max_depth': [3, 5, 10, 20, 50, None],
    'learning_rate': np.arange(.1, 1, .1),
    'gamma': np.arange(0, 1, .2),
    'subsample': np.arange(.5, 1, .1),
    'colsample_bytree': np.arange(.5, 1, .1),
    'reg_alpha': np.arange(0, 1, .2),
    'reg_lambda': np.arange(0, 1, .2)
}

X,y=np.array(data["embeddings"]).reshape(-1,2048),labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

xgb_model=XGBClassifier()
rs=RandomizedSearchCV(estimator=xgb_model, 
                       param_distributions=param_distributions, 
                       cv=4, n_iter=100, random_state=95, 
                       scoring='f1_macro')

#print(np.array(data["embeddings"]).shape)
rs.fit(np.array(data["embeddings"]).reshape(-1,2048), labels)
recognizer=rs.best_estimator_
print(rs.best_params_)

'''
param ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


clf = lgb.LGBMClassifier(max_depth=-1, random_state=95, silent=True, metric='None', n_jobs=-1, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param, 
    n_iter=32,
    scoring='roc_auc',
    cv=3,
    refit=True,
    random_state=95,
    verbose=True)
gs.fit(X_train, y_train)
recognizer=gs.best_estimator_
print(gs.best_params_)'''

print('Test F1 score is '+str(f1_score(y_test,recognizer.predict(X_test),average='weighted')))


# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()