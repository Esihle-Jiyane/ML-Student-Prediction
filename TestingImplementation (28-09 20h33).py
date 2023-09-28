import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from numpy import asarray
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from google.colab import drive

# Access files in your Google Drive
drive.mount('/content/drive')

# Load your dataset (replace 'your_dataset.csv' with the actual dataset file)
data = pd.read_csv('/content/drive/My Drive/OULAD/studentInfo.csv')

# Assuming your dataset has features (X) and labels (y)
X = data.drop('final_result', axis=1)  # Replace 'target_column' with the actual label column name
y = asarray(data['final_result'])
#print(X, y)

#Check for any null values to clean data
null_vals = data.isnull().all(axis=1)
null_rows = data[null_vals]
print(null_rows, "\n")

#another way to check for null values in the dataset.
#print ("TOTAL COLUMNS WITH NULL VALUES IN EACH COLUMN")
#data.isna().sum()

from imblearn.over_sampling._smote.base import OrdinalEncoder
#DATA TRANSFORMATION USING LABEL ENCODER
#NB: There are 1111 NULL VALUES in the imd_band column that are assigned the value 10 during label encoding.
#Should I drop all those rows since they have missing values or do I use them with the assignment of the value 10 to the class? Which one is best practice?

#Transform Target Column Data Into Numerical Values
#encoder = OrdinalEncoder()
#y_encoded = encoder.fit_transform(y.reshape(-1, 1))

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_))) #for testing if done correctly

#Transform All Other Categorical Data Into Numerical Values
encoded_data = X.copy()
X_encoding = encoded_data[['code_module','code_presentation','gender','region','highest_education','imd_band','age_band','disability']].apply(LabelEncoder().fit_transform)

#Merge back the label encoded columns to the rest of the dataset
X_encoded = pd.concat([X['id_student'], X_encoding, X['studied_credits'], X['num_of_prev_attempts']], axis=1)

#PRINTING TO CHECK IF DATA TRANSFORMATION WAS SUCCESSFUL
print(X_encoded, "\n")
print("\nNumeric Results:\n", y_encoded)
print("\nLabel Mapping:", label_mapping)

#SOLVING CLASS IMBALANCE PROBLEM USING SMOTE
#Can be done as below or done using a dictionary method
#Dictionary method allows you to specify values for the variables in the target class but the value specified must be greater than the existing value of that class

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
idx, c=np.unique(y, return_counts=True)
sns.barplot(x=idx, y=c, ax=ax[0])

X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_encoded, y)
idx, c=np.unique(y_resampled, return_counts=True)
sns.barplot(x=idx, y=c, ax=ax[1])
plt.show()

print("Origanal Sample: ", np.unique(y, return_counts=True))
print("After Sampling using SMOTE: ", np.unique(y_resampled, return_counts=True))

#PLOTTING THE CORRELATION MATRIX OF THE DATASET (UNSPLIT DATA)
#You could split the data before you do the feature selection and in the case of our dataset, the selected features will be exactly the same
plt.figure(figsize=(10, 10))
correlated = X_encoded.corr()
sns.heatmap(correlated, annot=True, cmap=plt.cm.CMRmap)
plt.show()

#Feature Selection Using PEARSON CORRELATION
#My Features are not Highly Correlated, what do I do in that case?

AllFeatures = X_encoded.columns
#print("Original Features", AllFeatures, "\n")

def correlation(dataset, threshold):
  corr_columns = set()
  corr_matrix = dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if abs(corr_matrix.iloc[i, j]) > threshold:
        colname = corr_matrix.columns[i]
        corr_columns.add(colname)
  return corr_columns

cor_features = correlation(X_encoded, 0.1)
print(len(set(cor_features)))
print("Selected Features: ", cor_features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.4, random_state=42)
#print(X_train.shape , X_test.shape)

# Create the OneVsOneClassifier with a base classifier (SVC in this example)
base_classifier = SVC(kernel='linear', probability=True)
ovo_classifier = OneVsOneClassifier(base_classifier)
print("OvO Classifier Created")

# Train the OvO classifier on the training data
ovo_classifier.fit(X_train, y_train)
print("OvO Classifier Successfully Trained")

# Predict the class labels for the test data
y_prediction = ovo_classifier.predict(X_test)
print(y_prediction)

# Convert predicted labels back to original class labels
y_original = encoder.inverse_transform(y_prediction)
print (y_original)
print("Prediction done")

# MODEL PERFORMACE EVALUATION
print(classification_report(y_test, y_prediction, target_names=encoder.classes_))
print("Evaluation done")

# Create the OneVsRestClassifier with a base classifier (SVC in this example)

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#THIS IS A HARDCODED SCATTERPLOT WHERE DATA IS FED MANUALLY, re-attempt feeding data from dataset
# Define class labels and counts
class_labels = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
class_counts = [3024, 7052, 12361, 10156]

# Generate random data points within each class
data = pd.DataFrame()
for label, count in zip(class_labels, class_counts):
    np.random.seed(42)  # for reproducibility
    feature1 = np.random.normal(loc=np.random.randint(20, 80), scale=10, size=count)
    feature2 = np.random.normal(loc=np.random.randint(20, 80), scale=10, size=count)
    class_data = pd.DataFrame({'Feature1': feature1, 'Feature2': feature2, 'Class': label})
    data = pd.concat([data, class_data])
print(data)

# Create a scatterplot using Seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Feature1', y='Feature2', hue='Class', palette='viridis', alpha=0.6)
plt.title('Scatterplot of Multiclass Data')
plt.legend(title='Class')
plt.grid(True)
plt.show()
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#Feature Selection Using Chi-Square
#My Features are not Highly Correlated, what do I do in that case?
#In Chi2, we'll check the relationship of the variables(features) in relation to the target variable

chi_score = chi2(X_encoded, y_encoded)
chi_score #print the score

pvalues = pd.Series(chi_score[1])
pvalues.index = X_encoded.columns
pvalues.sort_values(ascending=True)

#Select FEATURES who's pvalue is less that 0.05 as best practice
#In this case it means the selected feature would be "id_student", "imd_band", "studied_credits"
#These are not as relevant to the student grades prediction as the features selection by PEARSON CORRELATION
#Also, the results vary when using the split data and unsplit data in this technique because we have continous values

