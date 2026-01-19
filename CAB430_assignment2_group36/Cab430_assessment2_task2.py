
### Correlation ###
# retrieve data from database, generate correlation values between all input numeric atteibutes 
# including boolean and Risk_infection and Covid19_positive

# Load packages.
import pyodbc
import pandas as pd

## Retrive data from database
# Connection string to connect to SQL Server named instance.
conn_str = pyodbc.connect(DRIVER="{ODBC Driver 17 for SQL Server}", 
                          SERVER="localhost",
                          DATABASE="COVID19_Survey", 
                          Trusted_Connection="yes")

input_query_1 = '''SELECT [Participant_ID], [Gender], [Age], [Height], [Weight], [Bmi], [BloodType], [Insurance], [Race] FROM Participant '''

input_query_2 = '''SELECT [Response_ID],[Smoking] ,[Contact_count],[House_count],[Public_transport_count],[Working],[Covid19_symptoms]
                    ,[Covid19_contact],[Asthma],[Kidney_disease],[Liver_disease],[Compromised_immune],[Heart_disease],[Lung_disease]
                    ,[Diabetes],[Hiv_positive],[Hypertension],[Other_chronic],[Nursing_home],[Health_worker] FROM Response '''

input_query_3 = ''' SELECT  [Survey_ID],[Date],[Participant],[Response],[Risk_infection_level],[Risk_infection],[Risk_mortality],[Covid19_positive] FROM Fact_survey '''
# Results from the query are returned to Python using the Pandas read_sql function
participant_data = pd.read_sql(input_query_1, conn_str)
response_data = pd.read_sql(input_query_2, conn_str)
survey_data = pd.read_sql(input_query_3, conn_str)

# Display the beginning of the data frame to verify it looks correct.
print(survey_data.head(n=5))

print('\n--Examine the data type--')
print(survey_data.info()) 

## Display statistics of variables in the dataset
import numpy as np

## CODE BELOW IS TO DETERMINE NUMERICAL AND CATEGORICAL COLUMNS IN DATASET ##

#Describe statistics of numerical variables
#print("\nSummary Statistics - all numerical variables in participants table")
#print(participant_data.describe(include=[np.number]))
#print("\nSummary Statistics - all numerical variables in response table")
#print(response_data.describe(include=[np.number]))

#Describe statistics of categorical variables
#print("\nSummary Statistics - all categorical variables in participants table")
#print(participant_data.describe(include=[object]))
#print("\nSummary Statistics - all categorical variables in response table")
#print(response_data.describe(include=[object]))

## confirmed that numerical variables in the participants table are:
# Height, weight and Bmi

## numerical in response table are:
# contact_count, House_count and Public_transport_count 

# these are on top of the boolean values in response:
# covid_19_symptoms, covid19_contact, asthma, kidney_diesease, liver_dieseeas, compromised_immune, heart_diseas, lung_disease,  
# Diabetes, Hiv_positive, hypertension, Other_chornic, Hursing_home, Health_worker


## to get the correlation values, have to merge the tables 
import seaborn as sns
import matplotlib.pyplot as plt

# merge survey_data with participant_data on Participant_ID
merged_df = survey_data.merge(participant_data, left_on='Participant', right_on='Participant_ID', how='inner')

#  merge result with response_data on Response_ID
merged_df = merged_df.merge(response_data, left_on='Response', right_on='Response_ID', how='inner')


# convert the booleans to int 
merged_df['Covid19_positive'] = merged_df['Covid19_positive'].astype(int)
list = ['Covid19_symptoms', 'Covid19_contact', 'Asthma', 'Kidney_disease', 'Liver_disease', 'Compromised_immune', 'Heart_disease', 'Lung_disease',  
'Diabetes', 'Hiv_positive', 'Hypertension', 'Other_chronic', 'Nursing_home', 'Health_worker']

for name in list:
    if name in merged_df.columns:
        merged_df[name] = merged_df[name].astype(int)
    

# keep only numeric columns
numeric_df = merged_df.select_dtypes(include=['number'])
#print("content of numeric_df\n")   #debugging
#for col in numeric_df.columns:
   # print(col)

# Drop unnecessary ID columns to avoid confusion
numeric_df.drop(columns=['Participant_ID', 'Response_ID', 'Participant', 'Response', 'Survey_ID'], inplace=True) 


## generate correlation values between all input numeric and Risk_infection
corr_infection = numeric_df.corr(method='spearman')['Risk_infection'].sort_values(ascending=False)

## generate correlation values between all input numeric and Covid19_positive
#print(merged_df['Covid19_positive'].dtype)
#print(merged_df['Covid19_positive'].unique())
corr_covid19 = numeric_df.corr(method='spearman')['Covid19_positive'].sort_values(ascending=False)


# display the results
print("\n Correlation with Risk_infection:\n", corr_infection)
print("\n Correlation with Covid19_positive:\n", corr_covid19) 



### Feature selection ###

## set up the initial number of attributes by referencing the correlation values 

## ANOVA ##
# get the input and target variables
X = numeric_df.drop(columns=['Risk_infection', 'Covid19_positive'])
Y1 = numeric_df['Risk_infection'].values
Y2 = numeric_df['Covid19_positive'].values


# Use method SelectKBest to select top-3 variables  based on Analysis of Variance (ANOVA) values
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

select1 = SelectKBest(score_func = f_classif, k=10)
select2 = SelectKBest(score_func=f_classif, k=10)

# Return the selected features
fit1 = select1.fit(X, Y1)
fit2 = select2.fit(X, Y2)

print("\n---- Feature selection for anova ----")

# Feature names
print('\n input Column names:', X.columns.tolist())
print('\n target column names:', ['Risk_infection', 'Covid19_positive'])

# get the scores for each target attributes and the selected feature names
print('\nScores for Risk_infection:', fit1.scores_)
print('Top 10 features for Risk_infection from ANOVA:', select1.get_feature_names_out())

print('\nScores for Covid19_positive:', fit2.scores_)
print('Top 10 features for Covid19_positive from ANOVA:', select2.get_feature_names_out())


## Fit to data and select the top-10 variables
X_new1 = select1.fit_transform(X, Y1)
X_new2 = select2.fit_transform(X, Y2)
# Generate a DataFrame for the transformed data with the selected variables
transformed_dataset_ANOVA_1 = pd.DataFrame(X_new1) # anova for risk infection
transformed_dataset_ANOVA_2 = pd.DataFrame(X_new2)
print("\nTransformed dataset shape using ANOVA for Risk_infection:", transformed_dataset_ANOVA_1.shape)
print("\nTransformed dataset shape using ANOVA for Covid19_positive:", transformed_dataset_ANOVA_2.shape)

## Chi-squared ##
# Use scoring method Chi-squared, chi2
from sklearn.feature_selection import chi2
select3 = SelectKBest(score_func=chi2, k=10) # chisq for risk infection
select4 = SelectKBest(score_func=chi2, k=10)

# Return the selected features
fit3 = select3.fit(X, Y1) # chisq for risk infection
fit4 = select4.fit(X, Y2)

print("\n---- Feature selection for chi-sqaure ----")

# get the scores for each target attributes and the selected feature names
print('\nScores for Risk_infection:', fit3.scores_)
print('Top 10 features for Risk_infection from CHI-SQAURED:', select3.get_feature_names_out())

print('\nScores for Covid19_positive:', fit4.scores_)
print('Top 10 features for Covid19_positive from CHI-SQAURED:', select4.get_feature_names_out())

# Fit to data and select the top-10 variables
X_new3 = select3.fit_transform(X, Y1) # chisq for risk infection
X_new4 = select4.fit_transform(X, Y2)
# Generate a DataFrame for the transformed data with the selected variables
transformed_dataset_Chi_1 = pd.DataFrame(X_new3) # use for pred and eval
transformed_dataset_Chi_2 = pd.DataFrame(X_new4)
print("\nTransformed dataset shape using chi-squared for Risk_infection:", transformed_dataset_Chi_1.shape)
print("\nTransformed dataset shape using chi-squared for Covid19_positive:", transformed_dataset_Chi_2.shape)


## Prediciton and evaluation ##

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import metrics  
from sklearn.model_selection import train_test_split

## Define a method to complete prediction
def prediction(dataset, target_name, target_value, prediction_algorithm):
    
    # Set the column name type to string, required by prediction models. 
    dataset.columns = dataset.columns.astype(str)
    # Add the target columns to the transformed dataset
    dataset[target_name] = target_value

    # Get all the columns
    columns = dataset.columns.tolist()

    # Filter the columns to remove ones we don't want to use in the training
    columns = [c for c in columns if c not in [target_name]]

    # Set the target
    target = target_name

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(dataset[columns], dataset[target], test_size=0.2, random_state=1) 

    # Initialize the model class.
    model = prediction_algorithm

    # Fit the model to the training data.
    model.fit(X_train, y_train)

    # Generate predictions for the test set.
    predictions = model.predict(X_test) 

    #print("Predictions:\n", predictions)
    #print("Ground-truth:\n", y_test.values)

    # Compute accuracy of the prediction 
    print("Accuracy: ",metrics.accuracy_score(y_test, predictions))
    

# Call the method to generate predictions

print("\n---- Prediction ----")

for i in [5,8,10,12,15]:
    
    # change for anova
    select_anova = SelectKBest(score_func = f_classif, k=i)
    X_new_anova = select_anova.fit_transform(X, Y1)
    transformed_dataset_ANOVA = pd.DataFrame(X_new_anova)

    # change for shi square
    select_chi = SelectKBest(score_func=chi2, k=i)
    X_new_chi = select_chi.fit_transform(X, Y1)
    transformed_dataset_Chi = pd.DataFrame(X_new_chi)
 
    ## Gaussian Naive Bayes 
    print("\nFeature selection: ANOVA, Prediction algorithm: Naive Bayes , for K = ",i )
    prediction(transformed_dataset_ANOVA, 'Risk_infection', Y1, GaussianNB())

    print("\nFeature selection: Chi-sqaure, Prediction algorithm: Naive Bayes, for K = ",i  )
    prediction(transformed_dataset_Chi, 'Risk_infection', Y1, GaussianNB())

   

    ## Decision tree

    print("\nFeature selection: ANOVA, Prediction algorithm: Decision Trees, for K = ",i )
    prediction(transformed_dataset_ANOVA, 'Risk_infection', Y1, tree.DecisionTreeClassifier(criterion='gini', random_state=0))

    print("\nFeature selection: Chi square, Prediction algorithm: Decision Trees, for K = ",i)
    prediction(transformed_dataset_Chi, 'Risk_infection', Y1, tree.DecisionTreeClassifier(criterion='gini', random_state=0))







