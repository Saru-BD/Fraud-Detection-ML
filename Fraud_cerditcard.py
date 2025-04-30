#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Imports
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb

# Set display options
pd.set_option('display.max_columns', 100)

# Configuration parameters
RFC_METRIC = 'gini'
NUM_ESTIMATORS = 100
NO_JOBS = 4
VALID_SIZE = 0.20
TEST_SIZE = 0.20
NUMBER_KFOLDS = 5
RANDOM_STATE = 2018
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 1000
VERBOSE_EVAL = 50

# Path to dataset
IS_LOCAL = True

if IS_LOCAL:
    PATH = r"C:\Users\bette\OneDrive\Desktop\BP"  # Local folder containing creditcard.csv
else:
    PATH = "../input"

# List files in the dataset directory
print("Files in dataset path:", os.listdir(PATH))

# Load the dataset
data_df = pd.read_csv(os.path.join(PATH, "creditcard.csv"))
print("Dataset loaded successfully. Shape:", data_df.shape)
data_df.head()


# In[5]:


data_df.hist(bins=30, figsize=(30, 30))


# In[20]:


# Check the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(data_df.head())

# Info about the dataset (types, non-null counts)
print("\nDataset Information:")
data_df.info()

# Check for missing values
print("\nMissing values in each column:")
print(data_df.isnull().sum())


# In[21]:


# Descriptive statistics of the dataset
print("\nDescriptive statistics of the dataset:")
print(data_df.describe())


# In[22]:


# Check the distribution of the target variable
print("\nTarget variable distribution (fraud vs non-fraud):")
print(data_df['Class'].value_counts())


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the class distribution (fraud vs non-fraud)
sns.countplot(x='Class', data=data_df)
plt.title('Class Distribution (Fraud vs Non-Fraud)')
plt.show()


# In[27]:


# Check for the class distribution
class_counts = data_df['Class'].value_counts()
print("\nClass distribution:")
print(class_counts)


# In[6]:


import plotly.figure_factory as ff
from plotly.offline import iplot

# Split 'Time' feature based on fraud class
class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

# Create histogram data and labels for both classes
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

# Create a density plot using Plotly
fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))

# Show the plot
iplot(fig, filename='dist_only')


# In[29]:


# Create a new column 'Hour' which is the hour of the transaction
data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))

# Group by hour and class, and aggregate the transaction statistics
tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()

# Convert the aggregated data to a DataFrame
df = pd.DataFrame(tmp)

# Rename columns for easier readability
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']

# Display the first few rows
df.head()


# In[30]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show()


# In[32]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show()


# In[33]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Average Amount of Transactions")
plt.show()


# In[34]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Max", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Max", data=df.loc[df.Class==1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show()


# In[35]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show()


# In[36]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show()


# In[37]:


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
sns.boxplot(ax=ax1, x="Class", y="Amount", hue="Class", data=data_df, palette="PRGn", showfliers=True)
sns.boxplot(ax=ax2, x="Class", y="Amount", hue="Class", data=data_df, palette="PRGn", showfliers=False)
plt.suptitle("Transaction Amounts by Class")
plt.tight_layout()
plt.show()


# In[38]:


class_0 = data_df.loc[data_df['Class'] == 0]['Amount']
class_1 = data_df.loc[data_df['Class'] == 1]['Amount']

print("Class 0 (Not Fraud):")
print(class_0.describe())
print("\nClass 1 (Fraud):")
print(class_1.describe())


# In[39]:


import plotly.graph_objs as go
from plotly.offline import iplot

fraud = data_df[data_df['Class'] == 1]

trace = go.Scatter(
    x=fraud['Time'],
    y=fraud['Amount'],
    mode="markers",
    name="Fraud Amount",
    marker=dict(color='red', line=dict(color='darkred', width=1), opacity=0.5),
    text=fraud['Amount']
)

layout = dict(
    title='Amount of Fraudulent Transactions Over Time',
    xaxis=dict(title='Time [s]', showticklabels=True),
    yaxis=dict(title='Amount'),
    hovermode='closest'
)

fig = dict(data=[trace], layout=layout)
iplot(fig, filename='fraud-amount')


# In[88]:


# Combined and cleaned-up correlation heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 12))
corr = data_df.corr()

sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    annot_kws={'size': 8},
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'shrink': 0.8},
    xticklabels=corr.columns,
    yticklabels=corr.columns
)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title("Correlation Heatmap of Credit Card Transaction Features", fontsize=16)
plt.tight_layout()
plt.show()


# In[41]:


sns.lmplot(x='V20', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
plt.title("V20 vs Amount")

sns.lmplot(x='V7', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
plt.title("V7 vs Amount")
plt.show()


# In[42]:


sns.lmplot(x='V2', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
plt.title("V2 vs Amount")

sns.lmplot(x='V5', y='Amount', data=data_df, hue='Class', fit_reg=True, scatter_kws={'s': 2})
plt.title("V5 vs Amount")
plt.show()


# In[43]:


# List of all features
var = data_df.columns.values

# Split data by class
t0 = data_df[data_df['Class'] == 0]
t1 = data_df[data_df['Class'] == 1]

# Plotting setup
sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8, 4, figsize=(20, 28))
fig.suptitle("Feature Distributions by Class", fontsize=20)

# Loop through each feature
for i, feature in enumerate(var):
    plt.subplot(8, 4, i + 1)
    sns.kdeplot(t0[feature], bw_adjust=0.5, label="Class = 0", fill=True)
    sns.kdeplot(t1[feature], bw_adjust=0.5, label="Class = 1", fill=True)
    plt.xlabel(feature, fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc='upper right', fontsize=6)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


# In[44]:


target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
              'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
              'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
              'Amount']


# In[45]:


train_df, test_df = train_test_split(data_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
train_df, valid_df = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True)


# In[46]:


clf = RandomForestClassifier(
    n_jobs=NO_JOBS,
    random_state=RANDOM_STATE,
    criterion=RFC_METRIC,
    n_estimators=NUM_ESTIMATORS,
    verbose=False
)

clf.fit(train_df[predictors], train_df[target].values)


# In[47]:


preds = clf.predict(valid_df[predictors])


# In[48]:


tmp = pd.DataFrame({
    'Feature': predictors,
    'Feature importance': clf.feature_importances_
}).sort_values(by='Feature importance', ascending=False)

plt.figure(figsize=(10, 5))
plt.title('Feature Importances from RandomForest', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance', data=tmp, palette='viridis')
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


# In[49]:


cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(5, 5))
sns.heatmap(cm,
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,
            fmt='d',
            linewidths=0.5,
            cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# In[50]:


from sklearn.metrics import roc_auc_score

# Calculate ROC-AUC Score
roc_auc_rf = roc_auc_score(valid_df[target].values, preds)
print(f"ROC-AUC Score (RandomForestClassifier): {roc_auc_rf:.4f}")


# In[51]:


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(
    random_state=RANDOM_STATE,
    algorithm='SAMME.R',
    learning_rate=0.8,
    n_estimators=NUM_ESTIMATORS
)


# In[52]:


clf.fit(train_df[predictors], train_df[target].values)


# In[53]:


preds = clf.predict(valid_df[predictors])


# In[54]:


tmp = pd.DataFrame({
    'Feature': predictors,
    'Feature importance': clf.feature_importances_
}).sort_values(by='Feature importance', ascending=False)

plt.figure(figsize=(10, 5))
plt.title('Feature Importances from AdaBoost', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance', data=tmp, palette='coolwarm')
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


# In[55]:


cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(5, 5))
sns.heatmap(cm,
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,
            fmt='d',
            linewidths=0.5,
            cmap="Blues")
plt.title('Confusion Matrix - AdaBoost', fontsize=14)
plt.show()


# In[56]:


roc_auc_ada = roc_auc_score(valid_df[target].values, preds)
print(f"ROC-AUC Score (AdaBoostClassifier): {roc_auc_ada:.4f}")


# In[57]:


from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.02,
    depth=12,
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    bagging_temperature=0.2,
    od_type='Iter',
    metric_period=VERBOSE_EVAL,
    od_wait=100,
    verbose=True
)


# In[58]:


clf.fit(train_df[predictors], train_df[target].values)


# In[59]:


preds = clf.predict(valid_df[predictors])


# In[60]:


tmp = pd.DataFrame({
    'Feature': predictors,
    'Feature importance': clf.feature_importances_
}).sort_values(by='Feature importance', ascending=False)

plt.figure(figsize=(10, 5))
plt.title('Feature Importances from CatBoost', fontsize=14)
s = sns.barplot(x='Feature', y='Feature importance', data=tmp, palette='crest')
s.set_xticklabels(s.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.show()


# In[62]:


cm = pd.crosstab(valid_df[target].values, preds, rownames=['Actual'], colnames=['Predicted'])

plt.figure(figsize=(5, 5))
sns.heatmap(cm,
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,
            fmt='d',
            linewidths=0.5,
            cmap="Blues")
plt.title('Confusion Matrix - CatBoost', fontsize=14)
plt.show()


# In[63]:


roc_auc_catboost = roc_auc_score(valid_df[target].values, preds)
print(f"ROC-AUC Score (CatBoostClassifier): {roc_auc_catboost:.4f}")


# In[64]:


import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Prepare the DMatrix datasets
dtrain = xgb.DMatrix(train_df[predictors], label=train_df[target].values)
dvalid = xgb.DMatrix(valid_df[predictors], label=valid_df[target].values)
dtest = xgb.DMatrix(test_df[predictors], label=test_df[target].values)

# Watchlist to track performance on train and validation
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eta': 0.039,
    'max_depth': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'verbosity': 1  # replaces deprecated 'silent' param
}


# In[65]:


model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=MAX_ROUNDS,
    evals=watchlist,
    early_stopping_rounds=EARLY_STOP,
    maximize=True,
    verbose_eval=VERBOSE_EVAL
)


# In[66]:


fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, height=0.8, title="Feature Importance (XGBoost)", ax=ax, color="green")
plt.tight_layout()
plt.show()


# In[67]:


preds = model.predict(dtest)


# In[68]:


roc_auc = roc_auc_score(test_df[target].values, preds)
print(f"ROC-AUC Score (XGBoost on test set): {roc_auc:.4f}")


# In[69]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 7,
    'max_depth': 4,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'min_split_gain': 0,
    'scale_pos_weight': 150,
    'nthread': 8,
    'verbose': 0
}


# In[ ]:





# In[74]:


import lightgbm as lgb

# Prepare the datasets as LightGBM Dataset objects
dtrain = lgb.Dataset(train_df[predictors].values, 
                     label=train_df[target].values,
                     feature_name=predictors)

dvalid = lgb.Dataset(valid_df[predictors].values,
                     label=valid_df[target].values,
                     feature_name=predictors)

# Now train the model using LightGBM's Dataset objects
evals_results = {}

model = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dvalid],
    valid_names=['train', 'valid'],
    num_boost_round=MAX_ROUNDS,
    callbacks=[
        early_stopping(stopping_rounds=2*EARLY_STOP),
        log_evaluation(period=VERBOSE_EVAL)
    ]
)


# In[75]:


fig, ax = plt.subplots(figsize=(8, 5))
lgb.plot_importance(model, height=0.8, title="Features Importance (LightGBM)", ax=ax, color="red")
plt.show()


# In[76]:


preds = model.predict(test_df[predictors])
roc_auc = roc_auc_score(test_df[target], preds)
print(f"ROC-AUC on test set: {roc_auc:.4f}")


# In[87]:


import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import gc

# Define cross-validation parameters
kf = KFold(n_splits=NUMBER_KFOLDS, random_state=RANDOM_STATE, shuffle=True)

oof_preds = np.zeros(train_df.shape[0])
test_preds = np.zeros(test_df.shape[0])
feature_importance_df = pd.DataFrame()
n_fold = 0

# Loop over each fold in KFold
for train_idx, valid_idx in kf.split(train_df):
    train_x, train_y = train_df[predictors].iloc[train_idx], train_df[target].iloc[train_idx]
    valid_x, valid_y = train_df[predictors].iloc[valid_idx], train_df[target].iloc[valid_idx]

    model = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        num_leaves=31,  # Reduced from 80 to 31 to prevent overfitting
        max_depth=10,   # Limit the depth of trees
        colsample_bytree=0.98,
        subsample=0.78,
        reg_alpha=0.04,
        reg_lambda=0.073,
        subsample_for_bin=50,
        boosting_type='gbdt',
        is_unbalance=False,
        min_split_gain=0.1,  # Increased to require stronger splits
        min_child_weight=40,
        min_child_samples=510,
        objective='binary',
        metric='auc',
        silent=-1,
        min_data_in_leaf=20,  # Increased to avoid very small leaves
    )

    callbacks = [
        early_stopping(stopping_rounds=50),
        log_evaluation(period=50)
    ]

    model.fit(
        train_x, 
        train_y, 
        eval_set=[(train_x, train_y), (valid_x, valid_y)], 
        eval_metric='auc', 
        callbacks=callbacks
    )

    oof_preds[valid_idx] = model.predict_proba(valid_x, num_iteration=model.best_iteration_)[:, 1]
    test_preds += model.predict_proba(test_df[predictors], num_iteration=model.best_iteration_)[:, 1] / kf.n_splits

    fold_importance_df = pd.DataFrame({
        "feature": predictors,
        "importance": model.feature_importances_,
        "fold": n_fold + 1
    })
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    # Calculate the AUC for this fold
    fold_auc = roc_auc_score(valid_y, oof_preds[valid_idx])
    print(f"Fold {n_fold + 1} AUC : {fold_auc:.6f}")
    
    # Clear variables and collect memory
    del model, train_x, train_y, valid_x, valid_y
    gc.collect()
    n_fold += 1

# Final AUC score for out-of-fold predictions
full_auc = roc_auc_score(train_df[target], oof_preds)
print(f"Full OOF AUC score: {full_auc:.6f}")

# AUC score for the test data (average of all folds)
test_auc = roc_auc_score(test_df[target], test_preds)
print(f"The AUC score for the prediction from the test data was {test_auc:.6f}.")


# In[ ]:





# In[ ]:




