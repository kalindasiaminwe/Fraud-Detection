# -*- coding: utf-8 -*-
"""
Here we used AdaBoostClassifier algorithm and the GradientBoostingClassifier as a base estimator .
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

base_estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=0.005, max_depth=4, random_state=42)
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