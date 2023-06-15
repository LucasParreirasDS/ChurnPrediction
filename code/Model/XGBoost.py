import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def print_scores(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    pre_score = precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    f_score = f1_score(y_true, y_pred, average='weighted')
    return acc_score, pre_score, rec_score, f_score

data = pd.read_csv('data/raw/Churn_Modelling.csv')

df = data.copy()
    # df.drop(['RowNumber', 'CustomerId', 'Surname', 'Tenure', 'HasCrCard', 'EstimatedSalary'], axis=1, inplace=True)
df.drop(['RowNumber', 'CustomerId', 'Surname', 'EstimatedSalary'], axis=1, inplace=True)

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

# Split the data into training, validation, and test sets
x_cv, x_val, y_cv, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_cv, y_cv, test_size=0.25, random_state=42)  # 60% train, 20% validation, 20% test


# Define the scaler
scaler = StandardScaler()

# Scale the features using the training data
x_train = scaler.fit_transform(x_train)

# Apply the same scaling transformation to the validation and test data
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Define the XGBoost model with the best parameters
model = XGBClassifier(objective='binary:logistic', eval_metric='auc', max_depth=10, learning_rate=0.5, n_estimators=125)

kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
i = 1
 
for train_index, test_index in kfold.split(x_train, y_train):
    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
    
    # Train the model on the current fold
    model.fit(x_train_fold, y_train_fold)
    
    # Make predictions on the test/validation data
    y_test_pred = model.predict(x_test_fold)

    # Evaluate the model's performance on the test/validation data
    test_acc_score, test_pre_score, test_rec_score, test_f_score = print_scores(y_true=y_test_fold, y_pred=y_test_pred)

    # Print the test/validation metrics
    print(f"\n----- Test Metrics for fold {i}/10 -----")
    print(f"Accuracy: {(test_acc_score)*100:.2f}%")
    print(f"Precision: {(test_pre_score)*100:.2f}%")
    print(f"Recall: {(test_rec_score)*100:.2f}%")
    print(f"F1-Score: {(test_f_score)*100:.2f}%")
    
    i += 1



# Make predictions on the validation data
y_val_pred = model.predict(x_val)

# Evaluate the model's performance on the validation data
val_acc_score, val_pre_score, val_rec_score, val_f_score = print_scores(y_true=y_val, y_pred=y_val_pred)

# Print the test metrics
print('-'*30)
print("----- Validation Metrics -----")
print('-'*30)
print(f"Accuracy: {(val_acc_score)*100:.2f}%")
print(f"Precision: {(val_pre_score)*100:.2f}%")
print(f"Recall: {(val_rec_score)*100:.2f}%")
print(f"F1-Score: {(val_f_score)*100:.2f}%")


# Calculate and plot the confusion matrix for the test data
cm = confusion_matrix(y_true=y_val, y_pred=y_val_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Data)")
plt.show()
