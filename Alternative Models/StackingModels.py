# -*- coding: utf-8 -*-
"""
In this code, we define two base models: a RandomForestClassifier and a GradientBoostingClassifier.
We then define an AdaBoostClassifier as our meta model.
We use the StackingClassifier class to combine the predictions of the two base models and use them as input to the AdaBoostClassifier. Finally, we fit the stacked model on our training data and make predictions on the test set.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,  StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict



# define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
]

# define meta model
meta_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# create stacked model
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# fit the stacked model
stacked_model.fit(x_train, y_train)

# predict on test set
y_pred = stacked_model.predict(test)

boosting_preds = pd.DataFrame(y_pred, columns=['target'])

boosting_preds.head()

submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': boosting_preds['target']
    }
)

submission.head()
submission.to_csv('submission.csv', index=False)