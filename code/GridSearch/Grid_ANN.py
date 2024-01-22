import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf


def build_clf(unit, input_shape, dropout_rate):
    model = Sequential()

    model.add(Dense(unit, activation='relu', input_shape=(input_shape,)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(unit, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(unit, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(no_classes, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(), tf.keras.metrics.AUC(), f1_metric])

    return model


# Custom F1-score metric
def f1_metric(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())

    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1


# Loading the dataset
data = pd.read_csv('data/raw/Churn_Modelling.csv')

df = data.copy()
df.drop(['RowNumber', 'CustomerId', 'Surname', 'Tenure', 'HasCrCard', 'EstimatedSalary'], axis=1, inplace=True)

# Preprocessing
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography', 'NumOfProducts'])

# Feature Selection and split target
x = df.copy()
y = x.pop('Exited')
y = y.values.ravel()

# Dividing the dataset into features/labels and train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

no_classes = 1  # Number of classes for the model output
n_features = x.shape[1]  # Number of features for the model input

# Scaling the data
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Applying SMOTE to the training data
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)

# Creating the model with KFold and GridSearch
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf = KerasClassifier(build_fn=build_clf, epochs=500, input_shape=n_features, dropout_rate=0.2, verbose=1)

params = {'unit': [128, 256, 364],
          }

grid = GridSearchCV(estimator=clf, param_grid=params, cv=kfold, scoring='f1', verbose=3)
grid_result = grid.fit(x_train, y_train)

# Getting the best parameters
best_params = grid_result.best_params_
print("Best Parameters:", best_params)

# Evaluating the best model on the test set
best_model = grid_result.best_estimator_.model
y_pred = best_model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert float predictions to binary labels

accuracy = best_model.evaluate(x_test, y_test)[1]
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# Plotting training progress
history = grid_result.cv_results_
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.lineplot(data=history['mean_test_score'], ax=ax[0])
ax[0].set_title('Mean Test Score per Hyperparameter Combination')
ax[0].set_xlabel('Hyperparameter Combination')
ax[0].set_ylabel('Mean Test Score')

sns.lineplot(data=history['std_test_score'], ax=ax[1])
ax[1].set_title('Standard Deviation of Test Score per Hyperparameter Combination')
ax[1].set_xlabel('Hyperparameter Combination')
ax[1].set_ylabel('Standard Deviation')

plt.show()
