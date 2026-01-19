-- setting python into SQL
EXEC sp_configure  'external scripts enabled', 1 
RECONFIGURE WITH OVERRIDE 


EXEC sp_configure  'external scripts enabled' 

GO

USE COVID19_Survey

GO

DROP PROCEDURE IF EXISTS create_covid19_model_decisiontree;
GO

-- create stored procedure to train a decision tree model for COVID19_positive --
CREATE PROCEDURE create_covid19_model_decisiontree (@train_data NVARCHAR(MAX))
AS
BEGIN
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = N'
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# split X and Y
X = my_input_data.drop(columns=["Covid19_positive"])
y = my_input_data["Covid19_positive"]

# train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# evaluate model
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)

# serialize model to binary
model_binary = pickle.dumps(model)

# return  model name and serialized model as per ML table format
OutputDataSet = pd.DataFrame([
    ["DecisionTree_model", model_binary]
], columns=["model_name", "model"])
',
        @input_data_1 = @train_data,
        @input_data_1_name = N'my_input_data',
        @output_data_1_name = N'OutputDataSet'
    WITH RESULT SETS (("model_name" NVARCHAR(100), "model" VARBINARY(MAX)));

END;

GO 


-- declare input variables for decision tree 
-- features selected based of ANOVA results from task 2 
DECLARE @train_data NVARCHAR(MAX);
SET @train_data = '
SELECT Covid19_positive,Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Nursing_home
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
WHERE f.Date BETWEEN ''2020-05-01'' AND ''2020-07-31''
';

INSERT INTO [dbo].[COVID19_ML_models] -- store model into table 
EXEC create_covid19_model_decisiontree @train_data;

GO

-- clear COVID19_ML_models table IF NEEDED  
DELETE [dbo].[COVID19_ML_models]
GO

-- create stored procedure to train a Naive Bayes model to predict COVID19_positive --
DROP PROCEDURE IF EXISTS create_covid19_model_gaussiannb;
GO

CREATE PROCEDURE create_covid19_model_gaussiannb (@train_data NVARCHAR(MAX))
AS
BEGIN
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = N'
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Prepare data
X = my_input_data.drop(columns=["Covid19_positive"])
y = my_input_data["Covid19_positive"]

# train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Serialize model to binary
model_binary = pickle.dumps(model)

# Prepare output dataset
OutputDataSet = pd.DataFrame([
    ["GaussianNB_model", model_binary]
], columns=["model_name", "model"])
        ',
        @input_data_1 = @train_data,
        @input_data_1_name = N'my_input_data',
        @output_data_1_name = N'OutputDataSet'
    WITH RESULT SETS (
        ("model_name" NVARCHAR(100), "model" VARBINARY(MAX))
    );
END;

GO

-- declare input variables for GaussianNB, using same features selected for decision tree 
DECLARE @train_data NVARCHAR(MAX);
SET @train_data = '
SELECT Covid19_positive, Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Nursing_home
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
WHERE f.Date BETWEEN ''2020-05-01'' AND ''2020-07-31''
';

INSERT INTO COVID19_ML_models -- store model into table 
EXEC create_covid19_model_gaussiannb @train_data;

GO 

-- Display contents of the ML table to verify it works
SELECT * FROM [dbo].[COVID19_ML_models];

GO 

-- create stored procedure for predicting with decision tree -- 
DROP PROCEDURE IF EXISTS predict_covid19_decisiontree;
GO

CREATE PROCEDURE predict_covid19_decisiontree (@test_data NVARCHAR(MAX))
AS
BEGIN
    DECLARE @model VARBINARY(MAX);

    -- Load the saved decision tree model
    SELECT TOP 1 @model = model
    FROM COVID19_ML_models
    WHERE model_name = 'DecisionTree_model';

    -- Run prediction
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = N'
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load model
model = pickle.loads(model_binary)

# Convert Covid19_positive to int so it is retained in numeric_df
my_input_data["Covid19_positive"] = my_input_data["Covid19_positive"].astype(int)

# keep only numeric columns(inlcudes boolean) 
#numeric_df = my_input_data.select_dtypes(include=["number"])

# Prepare data
X = my_input_data.drop(columns=["Covid19_positive"])
y = my_input_data["Covid19_positive"]

# Predict
predictions = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, predictions)

# Output predictions and evaluation
output_df = my_input_data.copy()
output_df["prediction"] = predictions
output_df["correct"] = (predictions == y)
output_df["accuracy"] = accuracy  # constant value through out table

OutputDataSet = output_df[["Covid19_positive", "prediction", "correct", "accuracy"]]
        ',
        @input_data_1 = @test_data,
        @input_data_1_name = N'my_input_data',
        @output_data_1_name = N'OutputDataSet',
        @params = N'@model_binary varbinary(max)',
        @model_binary = @model
    WITH RESULT SETS (
        ("Covid19_positive" INT, "prediction" INT, "correct" BIT, "accuracy" FLOAT)
    );
END;
GO


-- declare test data for decision tree model  
DECLARE @test_data NVARCHAR(MAX);
SET @test_data = '
SELECT Covid19_positive,Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Nursing_home
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
WHERE f.Date BETWEEN ''2020-04-01'' AND ''2020-04-30''
';

EXEC predict_covid19_decisiontree @test_data;
GO 

-- create stored procedure for predicting with Naive Bayes method -- 
DROP PROCEDURE IF EXISTS predict_covid19_GaussianNB;
GO

CREATE PROCEDURE predict_covid19_GaussianNB (@test_data NVARCHAR(MAX))
AS
BEGIN
    DECLARE @model VARBINARY(MAX);

    -- Load the saved decision tree model
    SELECT TOP 1 @model = model
    FROM COVID19_ML_models
    WHERE model_name = 'GaussianNB_model';

    -- Run prediction
    EXEC sp_execute_external_script
        @language = N'Python',
        @script = N'
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load model
model = pickle.loads(model_binary)

# Convert Covid19_positive to int so it is retained in numeric_df
my_input_data["Covid19_positive"] = my_input_data["Covid19_positive"].astype(int)

# keep only numeric columns(inlcudes boolean) 
#numeric_df = my_input_data.select_dtypes(include=["number"])

# Prepare data
X = my_input_data.drop(columns=["Covid19_positive"])
y = my_input_data["Covid19_positive"]

# Predict
predictions = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, predictions)

# Output predictions and evaluation
output_df = my_input_data.copy()
output_df["prediction"] = predictions
output_df["correct"] = (predictions == y)
output_df["accuracy"] = accuracy  # constant value thourghout table 

OutputDataSet = output_df[["Covid19_positive", "prediction", "correct", "accuracy"]]
        ',
        @input_data_1 = @test_data,
        @input_data_1_name = N'my_input_data',
        @output_data_1_name = N'OutputDataSet',
        @params = N'@model_binary varbinary(max)',
        @model_binary = @model
    WITH RESULT SETS (
        ("Covid19_positive" INT, "prediction" INT, "correct" BIT, "accuracy" FLOAT)
    );
END;
GO


-- declare test data for GaussianNB model 
DECLARE @test_data NVARCHAR(MAX);
SET @test_data = '
SELECT Covid19_positive, Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Nursing_home
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
WHERE f.Date BETWEEN ''2020-04-01'' AND ''2020-04-30''
';

EXEC predict_covid19_GaussianNB @test_data;
GO

-- prediction and evaluation -- 
-- First set of input attributes: remove Nursing_home and add Risk_infection
--  first set of input attributes with decision tree
DECLARE @test_data2 NVARCHAR(MAX);
SET @test_data2 = '
SELECT Covid19_positive, Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Risk_infection
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
WHERE f.Date BETWEEN ''2020-04-01'' AND ''2020-04-30''
';
EXEC predict_covid19_decisiontree @test_data2;

GO 

-- first  set of input attributes with GaussianNB
DECLARE @test_data2 NVARCHAR(MAX);
SET @test_data2 = '
SELECT Covid19_positive, Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Risk_infection
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
WHERE f.Date BETWEEN ''2020-04-01'' AND ''2020-04-30''
';

EXEC predict_covid19_GaussianNB @test_data2;
GO

-- second set of input attributes: same input as task 3 subquesiton 2 input attributes but no contraints on dates 
-- second set of input attributes with decision tree 
DECLARE @test_data3 NVARCHAR(MAX);
SET @test_data3 = '
SELECT Covid19_positive,Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Nursing_home
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
';

EXEC predict_covid19_decisiontree @test_data3;
GO 

-- second set of input attributes with GaussianNB
DECLARE @test_data4 NVARCHAR(MAX);
SET @test_data4 = '
SELECT Covid19_positive,Height, Weight, Bmi, Contact_count, House_count, Covid19_symptoms, Covid19_contact, Kidney_disease, Compromised_immune, Nursing_home
FROM Fact_survey f
JOIN Response r ON f.Response = r.Response_ID
JOIN Participant p ON f.Participant = p.Participant_ID
';

EXEC predict_covid19_GaussianNB @test_data4;
GO 