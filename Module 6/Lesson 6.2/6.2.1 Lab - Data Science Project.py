# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Science Project
# MAGIC
# MAGIC **Objective**: *Design, complete, and assess a common data science project.*
# MAGIC
# MAGIC In this lab, you will use the data science process to design, build, and assess a common data science project.

# COMMAND ----------

# MAGIC %run "../../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Project Details
# MAGIC
# MAGIC In recent months, our health tracker company has noticed that many customers drop out of the sign-up process when they have to self-identify their exercise lifestyle (`ht_users.lifestyle`) – this is especially true for those with a "Sedentary" lifestyle. As a result, the company is considering removing this step from the sign-up process. However, the company knows this data is valuable for targeting introductory exercises and they don't want to lose it for customers that sign up after the step is removed.
# MAGIC
# MAGIC In this data science project, our business stakeholders are interested in identifying which customers have a sedentary lifestyle – specifically, they want to know if we can correctly identify whether somebody has a "Sedentary" lifestyle at least 95 percent of the time. If we can meet this objective, the organization will be able to remove the lifestyle-specification step of the sign-up process *without losing the valuable information provided by the data*.
# MAGIC
# MAGIC
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> There are no solutions provided for this project. You will need to complete it independently using the guidance detailed below and the previous labs from the project.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exercise 1
# MAGIC
# MAGIC Summary: 
# MAGIC * Specify the data science process question. 
# MAGIC * Indicate whether this is framed as a supervised learning or unsupervised learning problem. 
# MAGIC * If it is supervised learning, indicate whether the problem is a regression problem or a classification problem.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** When we are interested in predicting something, we are usually talking about a supervised learning problem.

# COMMAND ----------

##Answer: The data science question is: given a particular user's data, can we correctly classify them as having a "Sendentary" or "Non-Sendentary" lifestyle at least 95% of the time? Since we will be working with labeled data (i.e. the "lifestyle" feature) and we are trying to assign a classification to each user, this is a supervised learning/classification problem.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exercise 2
# MAGIC
# MAGIC Summary: 
# MAGIC
# MAGIC * Specify the data science objective. 
# MAGIC * Indicate which evaluation metric should be used to assess the objective.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Remember, the data science objective needs to be measurable.

# COMMAND ----------

##Answer: The data science objective is to train and test a machine learning classification model that will input user data and classify his/her lifestyle as either "Sendentary" or "Non-Sendentary". We need our model's accuracy to be 0.95 minimum on both the training and test datasets.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exercise 3
# MAGIC
# MAGIC Summary:
# MAGIC * Design a baseline solution.
# MAGIC * Develop a baseline solution – be sure to split data between training for development and test for assessment.
# MAGIC * Assess your baseline solution. Does it meet the project objective? If not, use it as a threshold for further development.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Recall that baseline solutions are meant to be easy to develop.

# COMMAND ----------

# MAGIC %python
# MAGIC import numpy as np
# MAGIC
# MAGIC # Create the 'ht_users' df in Pandas...
# MAGIC ht_users_spark_df = spark.read.table("ht_users")
# MAGIC ht_users_pandas_df = ht_users_spark_df.toPandas()
# MAGIC
# MAGIC # Create the 'ht_agg' df in Pandas...
# MAGIC ht_agg_spark_df = spark.read.table("ht_agg1")
# MAGIC ht_agg_pandas_df = ht_agg_spark_df.toPandas()
# MAGIC
# MAGIC # Create column 'lifestyle_recode' that recodes the 'lifestyle' column
# MAGIC # to either "Sedentary" or "Non-Sedentary"
# MAGIC
# MAGIC # ht_users_pandas_df['lifestyle_recode'] = np.where(ht_users_pandas_df['lifestyle'] == "Sedentary", "Sedentary", "Non-Sedentary")
# MAGIC ht_users_pandas_df['lifestyle_recode'] = ht_users_pandas_df['lifestyle'].apply(lambda x: 0 if x == 'Sedentary' else 1)
# MAGIC
# MAGIC # Drop 'first_name' and 'last_name' and 'lifestyle' columns they don't
# MAGIC # contribute anything to the analysis...
# MAGIC ht_users_pandas_df.drop(['first_name','last_name','lifestyle'],axis=1,inplace=True)
# MAGIC
# MAGIC ht_users_pandas_df['lifestyle_recode'].value_counts()

# COMMAND ----------

# Create the X matrix and y vector...
X = ht_users_pandas_df[['device_id','country']]
y = ht_users_pandas_df['lifestyle_recode']
     

import pandas as pd
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

# Check the baseline percentages for 'y_train'

lifestyle_summary = pd.DataFrame(y_train.value_counts()).reset_index().rename(columns={"index": "lifestyle", "lifestyle_recode": "freq"})
lifestyle_summary['pct'] = (lifestyle_summary['freq'] / lifestyle_summary['freq'].sum()) * 100
lifestyle_summary

lifestyle_summary = pd.DataFrame(y_test.value_counts()).reset_index().rename(columns={"index": "lifestyle", "lifestyle_recode": "freq"})
lifestyle_summary['pct'] = (lifestyle_summary['freq'] / lifestyle_summary['freq'].sum()) * 100
lifestyle_summary

##The baseline model shows that, for both the train and test datasets, users with a "Sedentary" lifestyle comprise approximately 10% - 12% of the population.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Exercise 4
# MAGIC
# MAGIC Summary: 
# MAGIC * Design the machine learning solution, but do not yet develop it. 
# MAGIC * Indicate whether a machine learning model will be used. If so, indicate which machine learning model will be used and what the label/output variable will be.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Consider solutions that align with the framing you did in Exercise 1.

# COMMAND ----------

##We will use a decision tree model as our proposed solution, since it usually performs well on problems like this one. Our label will be 'lifestyle_coding'.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 5
# MAGIC
# MAGIC Summary: 
# MAGIC * Explore your data. 
# MAGIC * Specify which tables and columns will be used for your label/output variable and your feature variables.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Consider aggregating features from other tables.

# COMMAND ----------

# Let's code the 'country' column as 1="United States", 0="Canada"
ht_users_pandas_df['country_coding'] = ht_users_pandas_df['country'].apply(lambda x: 1 if x == 'United States' else 0)
ht_users_pandas_df.drop(['country'],axis=1,inplace=True)
ht_users_pandas_df.head()
 
ht_agg_pandas_df.head()

merged_df = ht_users_pandas_df.merge(ht_agg_pandas_df, on='device_id').drop(['lifestyle','device_id'], axis=1)
merged_df.head()

##The dataset for modeling will include features from both 'ht_users_pandas_df' and 'ht_agg_pandas_df' datasets through a merge (join) on the common feature of 'device_id'. The 'X' matrix will contain the features ['country_recode', 'mean_bmi', 'mean_active_heartrate', 'mean_resting_heartrate', 'mean_vo2', 'mean_steps']. The 'y' or label vector will be ['lifestyle_recode'].



# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 6
# MAGIC
# MAGIC Summary: 
# MAGIC * Prepare your modeling data. 
# MAGIC * Create a customer-level modeling table with the correct output variable and features. 
# MAGIC * Finally, split your data between training and test sets. Make sure this split aligns with that of your baseline solution.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Consider how to make the data split reproducible.

# COMMAND ----------

# MAGIC %python
# MAGIC y = merged_df['lifestyle_coding']
# MAGIC X = merged_df.drop(['lifestyle_coding'],axis=1)  
# MAGIC
# MAGIC # Create the train and test datasets and ensure that the data split is reproducible by including a 'random_state' attribute.
# MAGIC X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 7
# MAGIC
# MAGIC Summary: 
# MAGIC * Build the model specified in your answer to Exercise 4. 
# MAGIC * Be sure to use an evaluation metric that aligns with your specified objective.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** This evaluation metric should align with the one used in your baseline solution.

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()

# Train the decision tree model on X_train and y_train
dt_model.fit(X_train,y_train)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Exercise 8
# MAGIC
# MAGIC Summary: 
# MAGIC * Assess your model against the overall objective. 
# MAGIC * Be sure to use an evaluation metric that aligns with your specified objective.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** Remember that we assess our models against our test data set to ensure that our solutions generalize.
# MAGIC
# MAGIC <img alt="Hint" title="Hint" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.3em" src="https://files.training.databricks.com/static/images/icon-light-bulb.svg"/>&nbsp;**Hint:** If your solution doesn't meet the objective, consider tweaking the model and data used by the model until it does meet the objective.

# COMMAND ----------

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

y_train_predicted = dt_model.predict(X_train)
print("training accuracy: ", accuracy_score(y_train, y_train_predicted))
print("training confusion matrix: ", confusion_matrix(y_train, y_train_predicted))

y_test_predicted = dt_model.predict(X_test)
print("test accuracy: ", accuracy_score(y_test, y_test_predicted))
print("test confusion matrix: ", confusion_matrix(y_test, y_test_predicted))

# COMMAND ----------

# MAGIC %md
# MAGIC We can observe the decision tree classifier does a great job of predicting "Sedentary" vs. "Non-Sedentary" users based on the input features we specified. We achieved accuracies of 100% on both the train and test datasets, hence the business objective of exceeding 95% accuracy has been successfully achieved.
# MAGIC
# MAGIC Future work would include training this model on larger (i.e., > 1M observations) datasets to ensure that it continues to generalize well. We have demonstrated successful results on 3000 observations, however this is relatively a small number.

# COMMAND ----------

# MAGIC %md
# MAGIC After completing all of the above objectives, you should be ready to communicate your results. Move to the next video in the lesson for a description on that part of the project.
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>