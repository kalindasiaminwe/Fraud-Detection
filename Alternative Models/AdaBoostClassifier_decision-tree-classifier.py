# -*- coding: utf-8 -*-
"""
AdaBoostClassifier is a machine learning algorithm used for classification tasks that is based on the concept of boosting.  
For this AdaBoostClassifier algorithm we used the DecisionTreeClassifier as a base estimator,
and trained multiple copies of it on different subsets of the data. The n_estimators parameter controls the number of estimators used in the ensemble, with a higher number generally leading to a more accurate and robust model.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

base_estimator = DecisionTreeClassifier(max_depth=3)
boosting_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=10, random_state=42)
boosting_model.fit(x_train, y_train)

boosting_preds = boosting_model.predict(test)
boosting_preds = pd.DataFrame(boosting_preds, columns=['target'])

boosting_preds.head()

submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': boosting_preds['target']
    }
)

submission.head()
submission.to_csv('submission.csv', index=False)