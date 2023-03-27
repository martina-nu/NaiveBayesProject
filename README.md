# Machine Learning-Based Sentiment Analysis of Google Play Store Reviews Using a Naive Bayes Classifier

## Objective:
This is a simple project that uses the Naive Bayes Classifier and Scikit-learn to create a Google Play Store review classifier with Python. The objective is to categorize user reviews as good or bad. 

In this dataset, we will use the 23 most popular mobile apps.

### Step 1:
We have three columns: package name, review, and polarity (0 = bad, 1 = good). Preprocess the data by eliminating the package name column and converting all reviews to lower case.

### Step 2:
Separate the target from the feature, and split your data.

### Step 3:
Vectorize your features and use Naive Bayes to classify the reviews as good or bad. We will not focus on hyper-tuning our model this time.

### Step 4:
Use app.py to create your pipeline. Save your Naive Bayes classification model in the 'models' folder.
