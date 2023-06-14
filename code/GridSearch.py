import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler    
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

data = pd.read_csv('data/raw/Churn_Modelling.csv')

df = data.copy()
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Preprocessing
 # Encoding our categorical features. We will use Labeling Encoder for the binaries and One Hot Encoding for the multi labels
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df = pd.get_dummies(df, columns=['Geography', 'NumOfProducts'])

# Feature Selection and split target
x = df[['NumOfProducts_2', 'NumOfProducts_1', 'Age', 'NumOfProducts_3', 'IsActiveMember', 'Geography_Germany']]
y = df['Exited']

y = y.values.ravel()

# Applying cross-validation
kfold = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=0)
x, y = smote.fit_resample(x, y)

# Define a instância do XGBClassifier
model = XGBClassifier(objective='binary:logistic', eval_metric='auc')


# Define your X and y variables

# Create a pipeline with MinMaxScaler and XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(objective='binary:logistic', eval_metric='auc'))
    ])

# Define the parameter grid for GridSearchCV
param_grid = {
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.1, 0.3, 0.5],
    'xgb__n_estimators': [50, 100, 150]
}

# Create the GridSearchCV object with StratifiedKFold
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='recall', verbose=3)

# Fit the GridSearchCV object to your data
grid_search.fit(x, y)

# Print the best parameters and best score
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
