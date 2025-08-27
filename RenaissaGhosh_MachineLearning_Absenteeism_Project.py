import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:\Users\rathi\course8-Renaissa\DS1_C9_S7_Project_AbsenteeismAtWork_Data.csv', sep=';')
data

data.dtypes

for i in data.columns:
    print(data[i].unique())

data.columns

categ = ['Reason for absence', 'Month of absence', 'Day of the week', 'Seasons', 'Education']
for i in categ:
    data[i] = data[i].astype("category")
    print(data[i])

data.isnull().sum()

# Task 2 (Part 1): Absenteeism Patterns – Demographics and Work Characteristics

# Set up subplot grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Absenteeism vs. Having a Son
sns.barplot(ax=axes[0, 0],
            x=data.groupby('Son')['Absenteeism time in hours'].mean().index,
            y=data.groupby('Son')['Absenteeism time in hours'].mean().values)
axes[0, 0].set_title('Absenteeism vs. Having a Son')
axes[0, 0].set_xlabel('Has Son (0=No, 1=Yes)')
axes[0, 0].set_ylabel('Average Absenteeism Hours')

# Absenteeism vs. Social Drinker
sns.barplot(ax=axes[0, 1],
            x=data.groupby('Social drinker')['Absenteeism time in hours'].mean().index,
            y=data.groupby('Social drinker')['Absenteeism time in hours'].mean().values)
axes[0, 1].set_title('Absenteeism vs. Social Drinker')
axes[0, 1].set_xlabel('Social Drinker (0=No, 1=Yes)')
axes[0, 1].set_ylabel('Average Absenteeism Hours')

# Absenteeism vs. Social Smoker
sns.barplot(ax=axes[1, 0],
            x=data.groupby('Social smoker')['Absenteeism time in hours'].mean().index,
            y=data.groupby('Social smoker')['Absenteeism time in hours'].mean().values)
axes[1, 0].set_title('Absenteeism vs. Social Smoker')
axes[1, 0].set_xlabel('Social Smoker (0=No, 1=Yes)')
axes[1, 0].set_ylabel('Average Absenteeism Hours')

# Absenteeism by Service Time
sns.lineplot(ax=axes[1, 1],
             x=data.groupby('Service time')['Absenteeism time in hours'].mean().index,
             y=data.groupby('Service time')['Absenteeism time in hours'].mean().values)
axes[1, 1].set_title('Absenteeism by Service Time')
axes[1, 1].set_xlabel('Service Time (Years)')
axes[1, 1].set_ylabel('Average Absenteeism Hours')

plt.tight_layout()
plt.show()

# Task 3 (Part 1): Frequency of Each Reason for Absence

# Count frequency of each reason
reason_counts = data['Reason for absence'].value_counts().sort_index()

# Plot the frequency distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=reason_counts.index, y=reason_counts.values)
plt.title('Frequency of Each Reason for Absence')
plt.xlabel('Reason for Absence (Code)')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Create a copy of the numeric data to avoid modifying the original
no_outlier = data.copy()

def outlier_cleanup(col):
    # Calculate IQR and outlier fences
    q1 = no_outlier[col].quantile(0.25)
    q3 = no_outlier[col].quantile(0.75)
    iqr = q3 - q1
    lf = q1 - (1.5 * iqr)  # lower fence
    uf = q3 + (1.5 * iqr)  # upper fence

    # Check column type for proper handling of integer and float types
    col_type = str(no_outlier[col].dtype)

    # Apply outlier treatment
    if col_type in ["int64", "int32", "int16", "int8"]:
        # For integer columns, round the fence values and cast them to integers
        no_outlier.loc[no_outlier[col] < lf, col] = int(round(lf))
        no_outlier.loc[no_outlier[col] > uf, col] = int(round(uf))
    else:
        # For float columns, directly assign the fence values
        no_outlier.loc[no_outlier[col] < lf, col] = lf
        no_outlier.loc[no_outlier[col] > uf, col] = uf

# Apply outlier cleanup to specific columns
my_conti_cols = ['Transportation expense', 'Distance from Residence to Work', 'Service time', 'Age', 'Work load Average/day ', 'Hit target', \
                 'Weight', 'Height', 'Body mass index', 'Absenteeism time in hours']
for col in my_conti_cols:
    outlier_cleanup(col)

no_outlier

# Label Encoding in categorical columns
from sklearn import preprocessing

categ = data.select_dtypes(exclude="number")

for i in categ.columns:
    encoder = preprocessing.LabelEncoder()  # Encoder is the object of 'LabelEncoder'.
    no_outlier[i] = encoder.fit_transform(no_outlier[i])

no_outlier

# Create new columns for high, moderate, and low absenteeism, and then change each column to integer values
no_outlier['High_absenteeism'] = no_outlier['Absenteeism time in hours'] > 7
no_outlier['High_absenteeism'] = no_outlier['High_absenteeism'].astype('int')
no_outlier['Moderate_absenteeism'] = (no_outlier['Absenteeism time in hours'] > 3) & (no_outlier['Absenteeism time in hours'] <= 7)
no_outlier['Moderate_absenteeism'] = no_outlier['Moderate_absenteeism'].astype('int')
no_outlier['Low_absenteeism'] = no_outlier['Absenteeism time in hours'] <= 3
no_outlier['Low_absenteeism'] = no_outlier['Low_absenteeism'].astype('int')
no_outlier

# Task 4 (Part 1): Predictive Analysis Using Multiple Linear Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # y = m1*x1 + m2*x2 + m3*x3 + m4*x4 + m5*x5 + c + e (Multiple Linear Regression)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def my_scaling(col):
    my_scaling_obj = StandardScaler()
    no_outlier[col] = pd.DataFrame(my_scaling_obj.fit_transform(no_outlier[col].to_numpy().reshape(-1,1)))
    return no_outlier[col]
my_numerical_cols = ['Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Son', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours']
for i in my_numerical_cols:
    no_outlier[i] = my_scaling(i)
no_outlier

no_outlier.dtypes

no_outlier.nunique()

my_corr = no_outlier.corr()[['High_absenteeism']].sort_values(by='High_absenteeism', ascending=False)
my_corr

# Check multi-colinearity between independent variables using the VIF score.

from statsmodels.stats.outliers_influence import variance_inflation_factor

no_abs = no_outlier.drop("High_absenteeism", axis=1)
no_abs

no_of_cols = no_abs.shape[1]
no_of_cols

def myVIF(x):
    vif = pd.Series(name="vif")
    for i in range(0, no_of_cols):
        vif[x.columns[i]] = variance_inflation_factor(x.values, i)
    return vif

vif_values = myVIF(no_abs)
print(vif_values)

# We should take the columns in which VIF scores are less than 10 (preferably less than 5).
# We are taking 4 independent columns: 'Reason for absence', 'Transportation expense', 'Distance from Residence to Work', and 'Absenteeism time in hours' (based on correlation with the target column and VIF score).

no_outlier.columns

my_cols = ['Reason for absence', 'Transportation expense', 'Distance from Residence to Work', 'Absenteeism time in hours']
for i in my_cols:
    sns.regplot(x=i, y="High_absenteeism", data=no_outlier, line_kws={"color":"black"})
    plt.show()

x_new = no_outlier[['Reason for absence', 'Transportation expense', 'Distance from Residence to Work', 'Absenteeism time in hours']].to_numpy().reshape(-1,4)
x_new

y = np.array(no_outlier["High_absenteeism"]).reshape(-1,1)
y

x_new_train, x_new_test, y_new_train, y_new_test = train_test_split(x_new, y, train_size = 0.7, random_state = 0)

# 'multi' is the name of the trained model.

# Step 1: Build a model using an algorithm.
# Step 2: Fit the model.
model = LinearRegression()
multi = model.fit(x_new_train, y_new_train)  # Here, we are building the model and training it in the same line.

# Step 3: Check training performance of the model.
r_square = multi.score(x_new_train, y_new_train)
print("R-square:", r_square)  # Training model performance is 72.7%, which is good. A bad model will not give good results.

# Step 4: Predict through unseen data using predict().
y_new_pred = multi.predict(x_new_test)  # We are predicting the values of y_test and storing it as 'y_prediction'. The input is given by passing the input as x_test -> if not matching, it is an error or residual (for numerical columns).
y_new_pred  # This is predicting and comparing with 'y_new_test' values.

mse = mean_squared_error(y_new_test, y_new_pred)
print("MSE:", mse)  # Smaller errors are better. Example: 2 is better than 2000 errors in case of MSE. MSE is numerical, not %, all errors are %.

# Step 5: Check testing (r_square) performance of the model. -> Checks the reliability of the model.
r2_new = r2_score(y_new_test, y_new_pred)
print("New R-squared:", r2_new)  # The model performance is 73%, which is still good.

# Linear regression works better when there is more than one input column.

multi.intercept_  # The linear regression model gives the intercept value.

multi.coef_  # Since we have taken 4 independent variables, we get 4 slopes for each line of best fit.

# Create a table containing residuals.
df_residual = pd.DataFrame()  # Creating an empty DataFrame
df_residual["y_pred"] = pd.DataFrame(y_new_pred)  # 'y_pred' - Creating a new column
# Assigning all 'y_new_pred' values to a 'y_pred' column in a DataFrame

df_residual["y_actual"] = pd.DataFrame(y_new_test)

# Residual = |Predicted - Actual|
df_residual["Residual"] = abs(df_residual["y_pred"] - df_residual["y_actual"])  # Error is calculated for each row.
df_residual  # Table with actual, predicted, and residual values

# Check if the assumption 'Homascedasticity' is applicable for this model.
sns.residplot(x = df_residual.index, y = "Residual", data = df_residual);

# According to the 'Residual' visual, homoscedasticity is followed.

# Assumption normality of residuals
sns.histplot(df_residual["Residual"], kde=True);

# This assumption fails because errors are not normally distributed.

len(no_outlier)

# Target column = 'High_absenteeism' is to be predicted. So, it will be considered as a dependent column.
# All other columns except for the column 'High_absenteeism' will be independent.

# We split the table into 4 parts that are called:
# x_train (training independent)
# x_test (testing independent)
# y_train (training dependent)
# y_test (testing dependent)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve

data['Absenteeism time in hours'].unique()

no_outlier['Absenteeism time in hours'].unique()

# Task 1 (Part 2): High Absenteeism

# Logistic Regression
# 1. Feature set
features = ['Reason for absence', 'Transportation expense', 'Distance from Residence to Work', 'Social drinker', 'Social smoker']
X = no_outlier[features]
y = no_outlier['High_absenteeism']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

# 3. Build and fit Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 4. Predict on test set
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)

# 5. Evaluation metrics
print("High Absenteeism")
print()
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob[:,1], pos_label = 1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

lr_probs = logreg.predict_proba(X_test)
ns_probs = [0 for _ in range(len(y_test))]  # We are using list comprehension.

# 6. ROC curve
lr_probs = lr_probs[:, 1]
# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# Summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# Plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# Add axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show the legend
plt.legend()
# Show the plot
plt.show()

# K-Nearest Neighbors
n = len(no_outlier)  # number of elements (rows) in a dataset without any outliers (input variable)
print(n)
# Assume the value 'k' as the square root of the number of elements (rows) in an array
k = int(round(n)**0.5)
print(k)  # This is NOT the best value of 'k'.

from sklearn.neighbors import KNeighborsClassifier
my_knn = KNeighborsClassifier(27)  # k-value needs to be selected by the user (k = 27).  # Step 1: Build the model.
my_model_1 = my_knn.fit(X_train, y_train)  # Train the model.
print(my_model_1.score(X_train, y_train))  # Step 2: Check the training performance.

y_pred = my_model_1.predict(X_test)
print(y_pred)

k2 = int(round(len(X_train)**0.5, 0))
print(k2)  # Initial but NOT final 'k'

# To find the value of 'k', we will create multiple temporary models.
accuracy = []
k_values = np.arange(4, 60, 2)
for my_k2 in k_values:
    temp = KNeighborsClassifier(my_k2)
    temp.fit(X_train, y_train)
    y2_pred = temp.predict(X_test)
    accuracy.append(accuracy_score(y_test, y2_pred))
print(accuracy)
print(k_values)

plt.plot(k_values, accuracy);

# By looking at the above plot, we will finalize the k-value as 23 to get maximum accuracy.
final_model = KNeighborsClassifier(k2)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print()
print("High Absenteeism")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Task 1 (Part 2): Moderate Absenteeism

# Logistic Regression
# 1. Feature set
features = ['Reason for absence', 'Transportation expense', 'Distance from Residence to Work', 'Social drinker', 'Social smoker']
X = no_outlier[features]
y = no_outlier['Moderate_absenteeism']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0)

# 3. Build and fit Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 4. Predict on test set
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)

# 5. Evaluation metrics
print("Moderate Absenteeism")
print()
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob[:,1], pos_label = 1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

lr_probs = logreg.predict_proba(X_test)
ns_probs = [0 for _ in range(len(y_test))]  # We are using list comprehension.

# 6. ROC curve
lr_probs = lr_probs[:, 1]
# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# Summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# Plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# Add axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show the legend
plt.legend()
# Show the plot
plt.show()

# K-Nearest Neighbors
n = len(no_outlier)  # number of elements (rows) in a dataset without any outliers (input variable)
print(n)
# Assume the value 'k' as the square root of the number of elements (rows) in an array
k = int(round(n)**0.5)
print(k)  # This is NOT the best value of 'k'.

from sklearn.neighbors import KNeighborsClassifier
my_knn = KNeighborsClassifier(27)  # k-value needs to be selected by the user (k = 27).  # Step 1: Build the model.
my_model_1 = my_knn.fit(X_train, y_train)  # Train the model.
print(my_model_1.score(X_train, y_train))  # Step 2: Check the training performance.

y_pred = my_model_1.predict(X_test)
print(y_pred)

k2 = int(round(len(X_train)**0.5, 0))
print(k2)  # Initial but NOT final 'k'

# To find the value of 'k', we will create multiple temporary models.
accuracy = []
k_values = np.arange(4, 60, 2)
for my_k2 in k_values:
    temp = KNeighborsClassifier(my_k2)
    temp.fit(X_train, y_train)
    y2_pred = temp.predict(X_test)
    accuracy.append(accuracy_score(y_test, y2_pred))
print(accuracy)
print(k_values)

plt.plot(k_values, accuracy);

# By looking at the above plot, we will finalize the k-value as 23 to get maximum accuracy.
final_model = KNeighborsClassifier(k2)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print()
print("Moderate Absenteeism")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Task 1 (Part 2): Low Absenteeism

# Logistic Regression
# 1. Feature set
features = ['Reason for absence', 'Disciplinary failure', 'Distance from Residence to Work', 'Social drinker', 'Social smoker']
X = no_outlier[features]
y = no_outlier['Low_absenteeism']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=0)

# 3. Build and fit Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 4. Predict on test set
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)

# 5. Evaluation metrics
print("Low Absenteeism")
print()
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, logreg.predict_proba(X_test)[:,1]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob[:,1], pos_label = 1)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

lr_probs = logreg.predict_proba(X_test)
ns_probs = [0 for _ in range(len(y_test))]  # We are using list comprehension.

# 6. ROC curve
lr_probs = lr_probs[:, 1]
# Calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# Summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# Calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# Plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# Add axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show the legend
plt.legend()
# Show the plot
plt.show()

# K-Nearest Neighbors
n = len(no_outlier)  # number of elements (rows) in a dataset without any outliers (input variable)
print(n)
# Assume the value 'k' as the square root of the number of elements (rows) in an array
k = int(round(n)**0.5)
print(k)  # This is NOT the best value of 'k'.

from sklearn.neighbors import KNeighborsClassifier
my_knn = KNeighborsClassifier(27)  # k-value needs to be selected by the user (k = 27).  # Step 1: Build the model.
my_model_1 = my_knn.fit(X_train, y_train)  # Train the model.
print(my_model_1.score(X_train, y_train))  # Step 2: Check the training performance.

y_pred = my_model_1.predict(X_test)
print(y_pred)

k2 = int(round(len(X_train)**0.5, 0))
print(k2)  # Initial but NOT final 'k'

# To find the value of 'k', we will create multiple temporary models.
accuracy = []
k_values = np.arange(4, 60, 2)
for my_k2 in k_values:
    temp = KNeighborsClassifier(my_k2)
    temp.fit(X_train, y_train)
    y2_pred = temp.predict(X_test)
    accuracy.append(accuracy_score(y_test, y2_pred))
print(accuracy)
print(k_values)

plt.plot(k_values, accuracy);

# By looking at the above plot, we will finalize the k-value as 23 to get maximum accuracy.
final_model = KNeighborsClassifier(k2)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print()
print("Low Absenteeism")
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Tasks 1-2 (Part 2): "Reason for absence" vs. "Absenteeism time in hours" Data

# K-Means Clustering
from sklearn.cluster import KMeans

X = np.array(no_outlier[["Reason for absence", "Absenteeism time in hours"]]).reshape(-1, 2)

err = []
my_cluster = np.arange(2, 11)
for k in my_cluster:
  temp = KMeans(n_clusters = k)
  temp.fit(X)
  err.append(temp.inertia_)  # '.inertia_' is provided to us in KMeans, it is the sum of squared distances of the samples

# We will create an elbow curve to visualize data and find the most appropriate number of custers.
plt.plot(my_cluster, err)
plt.xlabel("Value of K- number of clusters")
plt.ylabel("Value received from '.inertia_'")
plt.show()

# We will use other performance measures to see the appropriate number of clusters.
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

for j in my_cluster:
  temp_model = KMeans(n_clusters = j, random_state = 0)
  cluster_labels = temp_model.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels) * 100
  print("For clusters =", j,",","the silhouette score is", round(silhouette_avg,2))
  print("For clusters =", j,",","the davies bouldin score is", round(davies_bouldin_score(X, temp_model.labels_), 2))
  print("For clusters =", j,",","the calinski harabasz score is", round(calinski_harabasz_score(X, temp_model.labels_), 2))
  print()

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

range_n_clusters = range(2, 10)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical (v) line for average silhouette score of all the values
    sil_avg = silhouette_score(X, cluster_labels)
    print(sil_avg)
    ax1.axvline(x=sil_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# Silhouette curve can help us find the suitable number of clusters.

# Let's finalize k=3, means that we need 3 groups.
# Apply KMeans
final_km = KMeans(n_clusters=3, random_state=0)
final_km.fit(X)
final_centroid = final_km.cluster_centers_
final_labels = final_km.labels_  # our final model

# Select relevant features for clustering
clustering_features = ['Reason for absence', 'Absenteeism time in hours']
clustering_data = no_outlier[clustering_features]

# Optional: scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)
clusters = final_km.fit_predict(X_scaled)

# Assign cluster labels
clustering_data = clustering_data.copy()
clustering_data['cluster'] = clusters

# Plot the clusters
segments = ["Low Absenteeism", "Moderate Absenteeism", "High Absenteeism"]
colors = ["red", "yellow", "green"]

for i in range(3):
    plt.scatter(clustering_data.loc[clustering_data.cluster == i, "Reason for absence"],
                clustering_data.loc[clustering_data.cluster == i, "Absenteeism time in hours"],
                c=colors[i], label=segments[i])
plt.title("Employee Absenteeism Segmenation")
plt.xlabel("Reason for Absence")
plt.ylabel("Absenteeism Time (in hours)")
plt.legend()
plt.show()

# Tasks 1-2 (Part 2): "Disciplinary failure" vs. "Absenteeism time in hours" Data

# K-Means Clustering
from sklearn.cluster import KMeans

X = np.array(no_outlier[["Disciplinary failure", "Absenteeism time in hours"]]).reshape(-1, 2)

err = []
my_cluster = np.arange(2, 11)
for k in my_cluster:
  temp = KMeans(n_clusters = k)
  temp.fit(X)
  err.append(temp.inertia_)  # '.inertia_' is provided to us in KMeans, it is the sum of squared distances of the samples

# We will create an elbow curve to visualize data and find the most appropriate number of custers.
plt.plot(my_cluster, err)
plt.xlabel("Value of K- number of clusters")
plt.ylabel("Value received from '.inertia_'")
plt.show()

# We will use other performance measures to see the appropriate number of clusters.
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

for j in my_cluster:
  temp_model = KMeans(n_clusters = j, random_state = 0)
  cluster_labels = temp_model.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels) * 100
  print("For clusters =", j,",","the silhouette score is", round(silhouette_avg,2))
  print("For clusters =", j,",","the davies bouldin score is", round(davies_bouldin_score(X, temp_model.labels_), 2))
  print("For clusters =", j,",","the calinski harabasz score is", round(calinski_harabasz_score(X, temp_model.labels_), 2))
  print()

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

range_n_clusters = range(2, 10)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical (v) line for average silhouette score of all the values
    sil_avg = silhouette_score(X, cluster_labels)
    print(sil_avg)
    ax1.axvline(x=sil_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# Silhouette curve can help us find the suitable number of clusters.

# Let's finalize k=3, means that we need 3 groups.
# Apply KMeans
final_km = KMeans(n_clusters=3, random_state=0)
final_km.fit(X)
final_centroid = final_km.cluster_centers_
final_labels = final_km.labels_  # our final model

# Select relevant features for clustering
clustering_features = ['Disciplinary failure', 'Absenteeism time in hours']
clustering_data = no_outlier[clustering_features]

# Optional: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)
clusters = final_km.fit_predict(X_scaled)

# Assign cluster labels
clustering_data = clustering_data.copy()
clustering_data['cluster'] = clusters

segments = ["Low Absenteeism", "Moderate Absenteeism", "High Absenteeism"]
colors = ["red", "yellow", "green"]

n_clusters = 3
for i in range(n_clusters):
    plt.scatter(clustering_data.loc[clustering_data.cluster == i, "Disciplinary failure"],
                clustering_data.loc[clustering_data.cluster == i, "Absenteeism time in hours"],
                c=colors[i], label=segments[i])
plt.title("Employee Absenteeism Segmenation")
plt.xlabel("Disciplinary Failure")
plt.ylabel("Absenteeism Time (in hours)")
plt.legend()
plt.show()

# Tasks 1-2 (Part 2): "Distance from Residence to Work" vs. "Absenteeism time in hours" Data

# K-Means Clustering
from sklearn.cluster import KMeans

X = np.array(no_outlier[["Distance from Residence to Work", "Absenteeism time in hours"]]).reshape(-1, 2)

err = []
my_cluster = np.arange(2, 11)
for k in my_cluster:
  temp = KMeans(n_clusters = k)
  temp.fit(X)
  err.append(temp.inertia_)  # '.inertia_' is provided to us in KMeans, it is the sum of squared distances of the samples

# We will create an elbow curve to visualize data and find the most appropriate number of custers.
plt.plot(my_cluster, err)
plt.xlabel("Value of K- number of clusters")
plt.ylabel("Value received from '.inertia_'")
plt.show()

# We will use other performance measures to see the appropriate number of clusters.
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

for j in my_cluster:
  temp_model = KMeans(n_clusters = j, random_state = 0)
  cluster_labels = temp_model.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels) * 100
  print("For clusters =", j,",","the silhouette score is", round(silhouette_avg,2))
  print("For clusters =", j,",","the davies bouldin score is", round(davies_bouldin_score(X, temp_model.labels_), 2))
  print("For clusters =", j,",","the calinski harabasz score is", round(calinski_harabasz_score(X, temp_model.labels_), 2))
  print()

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

range_n_clusters = range(2, 10)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = clusterer.fit_predict(X)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical (v) line for average silhouette score of all the values
    sil_avg = silhouette_score(X, cluster_labels)
    print(sil_avg)
    ax1.axvline(x=sil_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# Silhouette curve can help us find the suitable number of clusters.

# Let's finalize k=3, means that we need 3 groups.
# Apply KMeans
final_km = KMeans(n_clusters=3, random_state=0)
final_km.fit(X)
final_centroid = final_km.cluster_centers_
final_labels = final_km.labels_  # our final model

# Select relevant features for clustering
clustering_features = ['Distance from Residence to Work', 'Absenteeism time in hours']
clustering_data = no_outlier[clustering_features]

# Optional: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)
clusters = final_km.fit_predict(X_scaled)

# Assign cluster labels
clustering_data = clustering_data.copy()
clustering_data['cluster'] = clusters

segments = ["Low Absenteeism", "Moderate Absenteeism", "High Absenteeism"]
colors = ["red", "yellow", "green"]

n_clusters = 3
for i in range(n_clusters):
    plt.scatter(clustering_data.loc[clustering_data.cluster == i, "Distance from Residence to Work"],
                clustering_data.loc[clustering_data.cluster == i, "Absenteeism time in hours"],
                c=colors[i], label=segments[i])
plt.title("Employee Absenteeism Segmenation")
plt.xlabel("Distance from Residence to Work")
plt.ylabel("Absenteeism Time (in hours)")
plt.legend()
plt.show()

# Tasks 1-2 (Part 2): "Social drinker" vs. "Absenteeism time in hours" Data

# K-Means Clustering
from sklearn.cluster import KMeans

X = np.array(no_outlier[["Social drinker", "Absenteeism time in hours"]]).reshape(-1, 2)

err = []
my_cluster = np.arange(2, 11)
for k in my_cluster:
  temp = KMeans(n_clusters = k)
  temp.fit(X)
  err.append(temp.inertia_)  # '.inertia_' is provided to us in KMeans, it is the sum of squared distances of the samples

# We will create an elbow curve to visualize data and find the most appropriate number of custers.
plt.plot(my_cluster, err)
plt.xlabel("Value of K- number of clusters")
plt.ylabel("Value received from '.inertia_'")
plt.show()

# We will use other performance measures to see the appropriate number of clusters.
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

for j in my_cluster:
  temp_model = KMeans(n_clusters = j, random_state = 0)
  cluster_labels = temp_model.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels) * 100
  print("For clusters =", j,",","the silhouette score is", round(silhouette_avg,2))
  print("For clusters =", j,",","the davies bouldin score is", round(davies_bouldin_score(X, temp_model.labels_), 2))
  print("For clusters =", j,",","the calinski harabasz score is", round(calinski_harabasz_score(X, temp_model.labels_), 2))
  print()

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

range_n_clusters = range(2, 10)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical (v) line for average silhouette score of all the values
    sil_avg = silhouette_score(X, cluster_labels)
    print(sil_avg)
    ax1.axvline(x=sil_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# Silhouette curve can help us find the suitable number of clusters.

# Let's finalize k=3, means that we need 3 groups.
# Apply KMeans
final_km = KMeans(n_clusters=3, random_state=0)
final_km.fit(X)
final_centroid = final_km.cluster_centers_
final_labels = final_km.labels_  # our final model

# Select relevant features for clustering
clustering_features = ['Social drinker', 'Absenteeism time in hours']
clustering_data = no_outlier[clustering_features]

# Optional: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)
clusters = final_km.fit_predict(X_scaled)

# Assign cluster labels
clustering_data = clustering_data.copy()
clustering_data['cluster'] = clusters

segments = ["Low Absenteeism", "Moderate Absenteeism", "High Absenteeism"]
colors = ["red", "yellow", "green"]

n_clusters = 3
for i in range(n_clusters):
    plt.scatter(clustering_data.loc[clustering_data.cluster == i, "Social drinker"],
                clustering_data.loc[clustering_data.cluster == i, "Absenteeism time in hours"],
                c=colors[i], label=segments[i])
plt.title("Employee Absenteeism Segmenation")
plt.xlabel("Social Drinker")
plt.ylabel("Absenteeism Time (in hours)")
plt.legend()
plt.show()

# Tasks 1-2 (Part 2): "Social smoker" vs. "Absenteeism time in hours" Data

# K-Means Clustering
from sklearn.cluster import KMeans

X = np.array(no_outlier[["Social smoker", "Absenteeism time in hours"]]).reshape(-1, 2)

err = []
my_cluster = np.arange(2, 11)
for k in my_cluster:
  temp = KMeans(n_clusters = k)
  temp.fit(X)
  err.append(temp.inertia_)  # '.inertia_' is provided to us in KMeans, it is the sum of squared distances of the samples

# We will create an elbow curve to visualize data and find the most appropriate number of custers.
plt.plot(my_cluster, err)
plt.xlabel("Value of K- number of clusters")
plt.ylabel("Value received from '.inertia_'")
plt.show()

# We will use other performance measures to see the appropriate number of clusters.
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

for j in my_cluster:
  temp_model = KMeans(n_clusters = j, random_state = 0)
  cluster_labels = temp_model.fit_predict(X)
  silhouette_avg = silhouette_score(X, cluster_labels) * 100
  print("For clusters =", j,",","the silhouette score is", round(silhouette_avg,2))
  print("For clusters =", j,",","the davies bouldin score is", round(davies_bouldin_score(X, temp_model.labels_), 2))
  print("For clusters =", j,",","the calinski harabasz score is", round(calinski_harabasz_score(X, temp_model.labels_), 2))
  print()

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

range_n_clusters = range(2, 10)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    # The vertical (v) line for average silhouette score of all the values
    sil_avg = silhouette_score(X, cluster_labels)
    print(sil_avg)
    ax1.axvline(x=sil_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

# Silhouette curve can help us find the suitable number of clusters.

# Let's finalize k=3, means that we need 3 groups.
# Apply KMeans
final_km = KMeans(n_clusters=3, random_state=0)
final_km.fit(X)
final_centroid = final_km.cluster_centers_
final_labels = final_km.labels_  # our final model

# Select relevant features for clustering
clustering_features = ['Social smoker', 'Absenteeism time in hours']
clustering_data = no_outlier[clustering_features]

# Optional: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustering_data)
clusters = final_km.fit_predict(X_scaled)

# Assign cluster labels
clustering_data = clustering_data.copy()
clustering_data['cluster'] = clusters

segments = ["Low Absenteeism", "Moderate Absenteeism", "High Absenteeism"]
colors = ["red", "yellow", "green"]

n_clusters = 3
for i in range(n_clusters):
    plt.scatter(clustering_data.loc[clustering_data.cluster == i, "Social smoker"],
                clustering_data.loc[clustering_data.cluster == i, "Absenteeism time in hours"],
                c=colors[i], label=segments[i])
plt.title("Employee Absenteeism Segmenation")
plt.xlabel("Social Smoker")
plt.ylabel("Absenteeism Time (in hours)")
plt.legend()
plt.show()

# 'fit_predict()':
# Fits the model to the data, and
# Predicts which group (or cluster) each data point belongs to — all in one step
