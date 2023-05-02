# -*- coding: utf-8 -*-
"""
BaggingClassifier is a machine learning algorithm used for classification tasks
that is based on the concept of bootstrap aggregating, or "bagging" for short.
It is an ensemble learning method that combines multiple base classifiers to create a more accurate and robust model.
"""

from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier()

bagging_model = BaggingClassifier(base_estimator=mlp_model, n_estimators=10, random_state=42)
bagging_model.fit(x_train, y_train)

bagging_preds = bagging_model.predict(test)

bagging_preds = pd.DataFrame(bagging_preds, columns=['target'])
bagging_preds.head()

submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': bagging_preds['target']
    }
)

submission.head()
submission.to_csv('submission.csv', index=False)