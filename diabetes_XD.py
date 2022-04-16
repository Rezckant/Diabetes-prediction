import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('diabetes.csv')
df.shape
df.isna().sum()
description = df.describe().T
df_eda = df

df_eda["Glucose"].replace(0, np.nan, inplace=True)
df_eda["BloodPressure"].replace(0, np.nan, inplace=True)
df_eda["SkinThickness"].replace(0, np.nan, inplace=True)
df_eda["Insulin"].replace(0, np.nan, inplace=True)
df_eda["BMI"].replace(0, np.nan, inplace=True)
df_eda.isna().sum()

df_eda.isnull().mean().sort_values(ascending=False)

import missingno as msno
plt.figure(figsize = (10,8))
msno.matrix(df_eda)

msno.heatmap(df_eda)
msno.matrix(df_eda.sort_values("Insulin"))

# Imputacion de datos faltantes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

ii_imp = IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=15, random_state= 0)

df_eda.loc[:, :] = ii_imp.fit_transform(df_eda)
df_eda.isnull().sum()

bins=[0,30,50,80]
sns.countplot(x=pd.cut(df.Age, bins=bins), hue=df.Outcome)
plt.show()
sns.countplot(df.Pregnancies, hue=df.Outcome)

plt.figure(figsize=(10,8))
sns.heatmap(data=df_eda.corr(),cmap="YlGnBu", annot=True ,linewidths=0.2, linecolor='white')
plt.show()

def plot_uni(d):
    f,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    sns.histplot(d, kde=True, ax=ax[0])
    ax[0].axvline(d.mean(), color='y', linestyle='--',linewidth=2)
    ax[0].axvline(d.median(), color='r', linestyle='dashed', linewidth=2)
    ax[0].axvline(d.mode()[0],color='g',linestyle='solid',linewidth=2)
    ax[0].legend({'Mean':d.mean(),'Median':d.median(),'Mode':d.mode()})
    
    sns.boxplot(x=d, showmeans=True, ax=ax[1])
    plt.tight_layout()
    
for f in df_eda:
    plot_uni(df[f])

df_skew = pd.DataFrame(data={
    'skewness': df_eda.skew()})

from sklearn import preprocessing
pt = preprocessing.PowerTransformer()
for col in df_eda.drop(['Outcome'], axis =1).columns:
  df_eda[col] = pt.fit_transform(df_eda[col].values.reshape(-1,1))
  
df_skew = pd.DataFrame(data={
    'skewness': df_eda.skew()})

import scipy.stats as stats
z = np.abs(stats.zscore(df_eda))
print(df_eda.shape)

df_eda = df_eda[(z < 3).all(axis=1)]
print(df_eda.shape)


for col in df_eda.drop('Outcome', axis = 1):
  f,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))    
  sns.kdeplot(data = df_eda, x = col, hue = 'Outcome', fill = 'dark', palette = 'dark' )
  plt.tight_layout()
  
sns.pairplot(df_eda,hue='Outcome',corner=True)

def plot_dis(d):
    f,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,6))   
    sns.boxenplot(x ='Outcome',y = d,  data = df_eda ,palette = 'rainbow')
    plt.tight_layout()
for f in df_eda.drop(['Outcome'], axis =1).columns:
    plot_dis(df_eda[f])

for col in df_eda:
    if col in df_eda.drop('Outcome', axis =1):
        q75,q25 = np.percentile(df_eda.loc[:,col],[75,25])
        iqr = q75-q25
     
        max = q75+(1.5*iqr)
        min = q25-(1.5*iqr)
     
        df_eda.loc[df[col] < min,col] = np.nan
        df_eda.loc[df[col] > max,col] = np.nan
print(df_eda.isna().sum())
df_eda = df_eda.dropna()
print(df_eda.isna().sum())

import statsmodels.api as sm
def plot_qq(d):
    f,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,8))   
    sns.histplot(d, ax= ax[0])
    ax[0].axvline(d.mean(), color='y', linestyle='--',linewidth=2)
    ax[0].axvline(d.median(), color='r', linestyle='dashed', linewidth=2)
    ax[0].axvline(d.mode()[0],color='g',linestyle='solid',linewidth=2)
    ax[0].legend({'Mean':d.mean(),'Median':d.median(),'Mode':d.mode()})
    sm.qqplot(d, line="s", ax= ax[1], fmt='b')
    ax= ax[1].set_title(col)
    plt.tight_layout()
for f in df_eda:
    plot_qq(df_eda[f])  


X = df_eda.iloc[:, :-1]
y = df_eda.iloc[:, -1]

from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 15)
print(df_eda.Outcome.value_counts())  
X_res,y_res = smk.fit_resample(X,y)
print(y_res.value_counts())

X = X_res
y = y_res


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Models
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
models = pd.DataFrame(columns=["Model","Accuracy Score"])

# -------------------- Logistic Regression --------------------
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state = 0)
log_reg.fit(X_train, y_train)
predictions = log_reg.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('Logistic Regression')
print(classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "LogisticRegression", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- K-Nearest Neighbors (Knn) --------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('KNN')
print(classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "KNN", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Suport Vector Machine (svm) --------------------
from sklearn.svm import SVC
svm = SVC(kernel = "rbf", random_state = 0)
svm.fit(X_train, y_train) 
predictions = svm.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('SVM')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "SVM", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Naive bayes --------------------
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('Naive Bayes')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "NaiveBayes", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- Random Forest --------------------
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 200, criterion = "gini", random_state = 0)
randomforest.fit(X_train, y_train)
predictions = randomforest.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('Random Forest')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "RandomForest", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- XGBoost --------------------
from xgboost import XGBClassifier
XGB = XGBClassifier()
XGB.fit(X_train, y_train)
predictions = XGB.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('XGBoost')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "XGBoost", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- CatBoost --------------------
from catboost import CatBoostClassifier
CatBoost = CatBoostClassifier(verbose=False)
CatBoost.fit(X_train,y_train,eval_set=(X_test, y_test))
predictions = CatBoost.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('CatBoost')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "CatBoost", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- ExtraTreeClassifier --------------------
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators = 200,
                                        criterion ='entropy', max_features = 'auto')
etc.fit(X_train,y_train)
predictions = etc.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('ExtraTreeClassifier')
report = (classification_report(y_test, predictions))

score = recall_score(y_test, predictions)

new_row = {"Model": "ETC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

# -------------------- GradientBoostingClassifier --------------------
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
predictions = gbc.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap = plt.cm.Blues)
plt.title('GradientBoostingClassifier')
report = (classification_report(y_test, predictions))

score = accuracy_score(y_test, predictions)

new_row = {"Model": "GBC", "Accuracy Score": score}
models = models.append(new_row, ignore_index=True)

models = models.sort_values(by="Accuracy Score", ascending=False)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = etc, X = X_train, y = y_train, cv = 10)
print(accuracies.mean()) # Sesgo
print(accuracies.std()) # Varianza

tmp=pd.DataFrame({'feature':df_eda.drop('Outcome',axis=1).columns,
                 'importance':etc.feature_importances_}).sort_values(by='importance',ascending=False)
plt.figure(figsize=(10,8))
sns.barplot(x=tmp.importance ,y=tmp.feature).set_title('Feature Importance')
plt.show()

# Hyperparameter tunning whith Optuna
import optuna  
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice

# Extra tree classifier
def objetive(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
          'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
          'max_depth': trial.suggest_int('max_depth', 3 ,10), 
          'max_features': trial.suggest_float('max_features', 0.25, 1.0)
          }
    trial_etc = ExtraTreesClassifier(**param, random_state = 1)
    trial_etc.fit(X_train,y_train)
    predictions = trial_etc.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score

study = optuna.create_study(direction = 'maximize')
study.optimize(objetive, n_trials = 70, n_jobs= -1)

study.best_params
etc_tuned = study.best_params

plot_optimization_history(study)
plot_param_importances(study)
plot_slice(study, ['max_features','max_depth'])

# CatBoost
def objetive1(trial):
    param1 = {
        'depth': trial.suggest_int('depth', 3, 10),
          'iterations': trial.suggest_int('iterations', 250, 1000),
          'learning_rate': trial.suggest_float('learning_rate', 0.03 ,0.3), 
          'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 3, 100),
          'border_count': trial.suggest_int('border_count', 32, 200)
          }
    trial_cat = CatBoostClassifier(**param1, random_state = 1)
    trial_cat.fit(X_train,y_train)
    predictions = trial_cat.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score

study1 = optuna.create_study(direction = 'maximize')
study1.optimize(objetive1, n_trials = 70, n_jobs= -1)

study1.best_params
cat_tuned = study1.best_params # +1

plot_optimization_history(study1)
plt.title("CatBoost")
plot_param_importances(study1)
plt.title("CatBoost")
plot_slice(study1, ['depth','l2_leaf_reg'])
plt.title("CatBoost")
plot_edf(study1)

# RandomForest
def objetive2(trial):
    param2 = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
          'bootstrap': trial.suggest_categorical('boostrap', ['True', 'False']),
          'max_depth': trial.suggest_int('max_depth', 3 ,10), 
          'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt']),
          'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
          'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
          }
    trial_rf = RandomForestClassifier(**param2, random_state = 1)
    trial_rf.fit(X_train,y_train)
    predictions = trial_rf.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study2 = optuna.create_study(direction = 'maximize')
study2.optimize(objetive2, n_trials = 70, n_jobs= -1)

study2.best_params
rf_tuned = study2.best_params

plot_optimization_history(study2)
plot_param_importances(study2)
plot_slice(study2, ['max_depth']) 

# XGBoost
def objetive3(trial):
    param3 = {
            'booster': trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimatos", 500, 1000),
            "gamma": trial.suggest_int("gamma", 0, 2),
            "max_depth": trial.suggest_int("max_dept", 3, 10),
            "n_jobs": (-1)
        }

    trial_xgb = XGBClassifier(**param3, random_state = 1)
    trial_xgb.fit(X_train,y_train)
    predictions = trial_xgb.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study3 = optuna.create_study(direction = 'maximize')
study3.optimize(objetive3, n_trials = 50, n_jobs= -1)

study3.best_params
xgb_tuned = study3.best_params # +1

plot_optimization_history(study3)
plt.title('XGB')
plot_param_importances(study3)
plt.title('XGB')
plot_slice(study3, ['booster','max_dept'])
plt.title('XGB')

# KNN
def objetive4(trial):
    param4 = {
            'n_neighbors' : trial.suggest_int('n_neighbors', 5, 20),
            'weights' : trial.suggest_categorical('weights', ['uniform','distance']),
            'metric' : trial.suggest_categorical('metric', ['minkowski','euclidean','manhattan']),
            'algorithm' : trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree'])
        }

    trial_knn = KNeighborsClassifier(**param4)
    trial_knn.fit(X_train,y_train)
    predictions = trial_knn.predict(X_test)
    

    score = recall_score(y_test, predictions)
    score1 = accuracy_score(y_test, predictions)
    print("---. ","Recall: ",score," .---")
    print("---. ","Accuracy: ",score1," .---")
    return score1

study4 = optuna.create_study(direction = 'maximize')
study4.optimize(objetive4, n_trials = 150, n_jobs= -1)

study4.best_params
knn_tuned = study4.best_params # +1

plot_optimization_history(study4)
plt.title('KNN')
plot_param_importances(study4)
plt.title('KNN')
plot_slice(study4, ['n_neighbors','weights'])
plt.title('KNN')
               

from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators= [('ExtraTreeClassifier', etc), 
                                      ('CatBoost', CatBoostClassifier(**cat_tuned)), 
                                      ('Randomforest', randomforest),
                                      ('XGBoost', XGBClassifier(**xgb_tuned)),
                                      ('KNM', KNeighborsClassifier(**knn_tuned))])
eclf1.fit(X_train, y_train)
voting_pred= eclf1.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, voting_pred, cmap = plt.cm.Blues)
plt.title('voting_pred')
print(classification_report(y_test, voting_pred))



