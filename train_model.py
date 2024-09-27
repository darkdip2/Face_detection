from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import argparse
import pickle



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
'''
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

xgb_model=XGBClassifier()
rs=RandomizedSearchCV(estimator=xgb_model, 
                       param_distributions=param_distributions, 
                       cv=4, n_iter=100, random_state=95, 
                       scoring='f1_macro')

#print(np.array(data["embeddings"]).shape)
rs.fit(np.array(data["embeddings"]).reshape(-1,2048), labels)
recognizer=rs.best_estimator_
print(rs.best_params_)'''

# Define the models
models = {
    'SVM': SVC(probability=True),
    'LR': LogisticRegression(max_iter=1000)
}

param_distributions = {
    'SVM': {
        'kernel': ['linear', 'rbf', 'poly']
    },
    'LR': {
        'C': [0.1, 1, 10],
        'max_iter': [100, 500, 1000]
    }
}

recognizer=None
for model in models:
    rs_model = models[model]
    rs = RandomizedSearchCV(estimator=rs_model, param_distributions=param_distributions[model], cv=4, n_iter=100, random_state=95, scoring='f1_macro')
    rs.fit(np.array(data["embeddings"]).reshape(-1,2048), labels)
    if model=='SVM':recognizer=rs.best_estimator_
    if model=='LR':recognizer=rs.best_estimator_
    print(f"Best parameters for {model}: {rs.best_params_}")


# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()