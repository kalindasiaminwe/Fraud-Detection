# -*- coding: utf-8 -*-
"""

#  Multilayer Perceptron (MLP)
MLPClassifier is a machine learning algorithm used for classification tasks,
and is a type of artificial neural network known as a Multilayer Perceptron (MLP).
The MLPClassifier is based on the concept of a neural network,
which is a collection of interconnected nodes, or neurons, that work together to learn patterns in the data.
"""

#MLP Classier
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(test)
y_pred = pd.DataFrame(y_pred, columns=['target'])
y_pred.head()

submission = pd.DataFrame(
    {
        'client_id': sub_client_id,
        'target': y_pred['target']
    }
)

submission.head()
submission.to_csv('submission.csv', index=False)