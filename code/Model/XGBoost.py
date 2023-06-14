import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

def print_scores(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    pre_score = precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred, average='weighted')
    return acc_score, pre_score, rec_score, f_score

data = pd.read_csv('data/raw/Churn_Modelling.csv')

df = data.copy()
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Preprocessing
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography', 'NumOfProducts'])

# Feature Selection and split target
x = df.copy()
y = x.pop('Exited')
y = y.values.ravel()

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=0)
x, y = smote.fit_resample(x, y)

# Define the scaler
scaler = MinMaxScaler()  # or StandardScaler()

# Scale the features
x = scaler.fit_transform(x)

# Applying cross-validation
kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

# Define the XGBoost model with the best parameters
model = XGBClassifier(
                       objective='binary:logistic', eval_metric='auc', max_depth=6, learning_rate=0.5, n_estimators=50)

metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

# Iterate over the folds using StratifiedKFold
for train_index, test_index in kfold.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model on the training data
    model.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(x_test)

    # Evaluate the model's performance on the current fold
    acc_score, pre_score, rec_score, f_score = print_scores(y_true=y_test, y_pred=y_pred)
    metrics['accuracy'].append(acc_score)
    metrics['precision'].append(pre_score)
    metrics['recall'].append(rec_score)
    metrics['f1_score'].append(f_score)

mean_metrics = {
    'accuracy': np.mean(metrics['accuracy']),
    'precision': np.mean(metrics['precision']),
    'recall': np.mean(metrics['recall']),
    'f1_score': np.mean(metrics['f1_score'])
}

print("Mean Metrics:")
for metric, value in mean_metrics.items():
    print(f"{metric}: {value}")
