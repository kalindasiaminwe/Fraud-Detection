# -*- coding: utf-8 -*-
"""

Random Forest Classifier is a machine learning algorithm used for classification tasks.
It is an ensemble learning method that combines multiple decision trees to create a more accurate and robust model.
"""

from sklearn.ensemble import RandomForestClassifier
RFC_METRIC = 'gini'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier

RANDOM_STATE = 2018

clf = RandomForestClassifier(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)

clf.fit(x_train, y_train)

preds = clf.predict(test)
preds = pd.DataFrame(preds, columns=['target'])
preds.head()

submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': preds['target']
    }
)

submission.head()
submission.to_csv('submission.csv', index=False)