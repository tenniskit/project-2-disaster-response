# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from timeit import default_timer as timer
from datetime import timedelta





def extract_category_names(df):
    filter_cols = ['index', 'id', 'message', 'original', 'genre']
    category_names = []
    for col in df.columns.values:
        if col not in filter_cols:
            category_names.append(col)
            
    return category_names


def tokenize(text):
    
    # Punctuation Removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data from database
engine = create_engine('sqlite:///C:/GoogleDrive/Teach/Udacity/Become a Data Scientist/Project_2/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)  


pd.set_option('display.max_columns', None)
print("df.shape=", df.shape)


category_names = extract_category_names(df)

print("category_names=", category_names)


X = df["message"].values
Y = df[category_names].values

print("X.shape=", X.shape, "Y.shape=", Y.shape)





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
rfc = RandomForestClassifier()
clf = MultiOutputClassifier(rfc)


start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:00:18.565973




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()


param_grid = { 
}

rfc = RandomForestClassifier()
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid)
clf = MultiOutputClassifier(CV_rfc)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:03:54.012655






""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()


param_grid = { 
}

rfc = RandomForestClassifier()
clf = MultiOutputClassifier(rfc)
cv = GridSearchCV(estimator=clf, param_grid=param_grid)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
cv.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
print("\nBest Parameters:", cv.best_params_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:03:54.675597





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()


param_grid = { 
    'estimator__n_estimators': [100,105] # It lowers the range of hyperparameters in order to prevent it from taking too long to run.
}

rfc = RandomForestClassifier()
pipeline = MultiOutputClassifier(rfc)
cv = GridSearchCV(estimator=pipeline, param_grid=param_grid)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
cv.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
print("\nBest Parameters:", cv.best_params_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:07:37.644530
# Best Parameters: {'estimator__n_estimators': 105}



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()


param_grid = { 
    'estimator__n_estimators': [100,105] # It lowers the range of hyperparameters in order to prevent it from taking too long to run.
}

rfc = RandomForestClassifier()
pipeline = MultiOutputClassifier(rfc)
cv = GridSearchCV(estimator=pipeline, param_grid=param_grid)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
cv.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
print("\nBest Parameters:", cv.best_params_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:48:58.974002
# Best Parameters: {'estimator__n_estimators': 105}





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()


param_grid = { 
    'estimator__n_estimators': [100,101] # It lowers the range of hyperparameters in order to prevent it from taking too long to run.
}

rfc = RandomForestClassifier()
pipeline = MultiOutputClassifier(rfc)
cv = GridSearchCV(estimator=pipeline, param_grid=param_grid)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
cv.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
print("\nBest Parameters:", cv.best_params_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 3:22:01.855443
# Best Parameters: {'estimator__n_estimators': 100}





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()



rfc = RandomForestClassifier()
pipeline = MultiOutputClassifier(rfc)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
pipeline.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:05:36.633374





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()


param_grid = { 
    'estimator__n_estimators': [100,101] # It lowers the range of hyperparameters in order to prevent it from taking too long to run.
}

rfc = RandomForestClassifier()
pipeline = MultiOutputClassifier(rfc)
cv = GridSearchCV(estimator=pipeline, param_grid=param_grid)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
cv.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
print("\nBest Parameters:", cv.best_params_)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 
# 




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
rfc = RandomForestClassifier()
pipeline = MultiOutputClassifier(rfc)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
pipeline.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 0:07:19.795098





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print("X_train.shape=", X_train.shape,
      "X_test.shape=", X_test.shape,
      "y_train.shape=", y_train.shape,
      "y_test.shape=", y_test.shape)


vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
rfc = RandomForestClassifier()
clf = MultiOutputClassifier(rfc)


param_grid = { 
    'estimator__n_estimators': [95, 100]
}

cv = GridSearchCV(estimator=clf, param_grid=param_grid)



start = timer()

# train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
cv.fit(X_train_tfidf, y_train)

# Takes time.....

end = timer()
print(timedelta(seconds=end-start))
print("\nBest Parameters:", cv.best_params_)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 3:54:44.839906
# Best Parameters: {'estimator__n_estimators': 95}


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# use model to predict classification for query
query = "we are more than 50 people on the street. Please help us find tent and food"

# predict on test data
X_test_counts = vect.transform([query])
X_test_tfidf = tfidf.transform(X_test_counts)
classification_labels = cv.predict(X_test_tfidf)

print("classification_labels.shape = ", classification_labels.shape)
print(classification_labels)


print("classification_labels[0].shape = ", classification_labels[0].shape)
print(classification_labels[0])

for i in range(len(category_names)):
    print(category_names[i], "=", classification_labels[0][i] )
    
"""
classification_labels.shape =  (1, 36)
[[1 1 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]]
classification_labels[0].shape =  (36,)
[1 1 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
related = 1
request = 1
offer = 0
aid_related = 1
medical_help = 0
medical_products = 0
search_and_rescue = 0
security = 0
military = 0
child_alone = 0
water = 0
food = 1
shelter = 1
clothing = 0
money = 0
missing_people = 0
refugees = 0
death = 0
other_aid = 0
infrastructure_related = 0
transport = 0
buildings = 0
electricity = 0
tools = 0
hospitals = 0
shops = 0
aid_centers = 0
other_infrastructure = 0
weather_related = 0
floods = 0
storm = 0
fire = 0
earthquake = 0
cold = 0
other_weather = 0
direct_report = 1
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

avg_accuracy = 0
for i in range(len(category_names)):
    accuracy = (classification_labels[:, i]==y_test[:, i]).mean()
    avg_accuracy = avg_accuracy + accuracy
    print(i, ": target_name=", category_names[i])
    print("Accuracy=" , accuracy)

avg_accuracy = avg_accuracy/36
print("Average Accuracy=" , avg_accuracy)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
0 : target_name= related
Accuracy= 0.7515255530129672
1 : target_name= request
Accuracy= 0.17009916094584288
2 : target_name= offer
Accuracy= 0.9946605644546148
3 : target_name= aid_related
Accuracy= 0.41418764302059496
4 : target_name= medical_help
Accuracy= 0.9212433257055682
5 : target_name= medical_products
Accuracy= 0.9496567505720824
6 : target_name= search_and_rescue
Accuracy= 0.9717772692601068
7 : target_name= security
Accuracy= 0.9834096109839817
8 : target_name= military
Accuracy= 0.9681540808543097
9 : target_name= child_alone
Accuracy= 1.0
10 : target_name= water
Accuracy= 0.9336384439359268
11 : target_name= food
Accuracy= 0.11003051106025935
12 : target_name= shelter
Accuracy= 0.08905415713196034
13 : target_name= clothing
Accuracy= 0.9832189168573608
14 : target_name= money
Accuracy= 0.9774980930587338
15 : target_name= missing_people
Accuracy= 0.9876048817696415
16 : target_name= refugees
Accuracy= 0.9624332570556827
17 : target_name= death
Accuracy= 0.9546147978642258
18 : target_name= other_aid
Accuracy= 0.8625095347063311
19 : target_name= infrastructure_related
Accuracy= 0.9334477498093059
20 : target_name= transport
Accuracy= 0.952326468344775
21 : target_name= buildings
Accuracy= 0.9490846681922197
22 : target_name= electricity
Accuracy= 0.9816933638443935
23 : target_name= tools
Accuracy= 0.9933257055682685
24 : target_name= hospitals
Accuracy= 0.988558352402746
25 : target_name= shops
Accuracy= 0.9948512585812357
26 : target_name= aid_centers
Accuracy= 0.9879862700228833
27 : target_name= other_infrastructure
Accuracy= 0.9553775743707094
28 : target_name= weather_related
Accuracy= 0.7278794813119756
29 : target_name= floods
Accuracy= 0.9181922196796338
30 : target_name= storm
Accuracy= 0.9088482074752098
31 : target_name= fire
Accuracy= 0.986651411136537
32 : target_name= earthquake
Accuracy= 0.9092295957284515
33 : target_name= cold
Accuracy= 0.9786422578184591
34 : target_name= other_weather
Accuracy= 0.9508009153318078
35 : target_name= direct_report
Accuracy= 0.18783371472158658
Average Accuracy= 0.8413901601830662
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pickle
model_filepath = "C:\\GoogleDrive\\Teach\\Udacity\\Become a Data Scientist\\Project_2\\release\\disaster_response_pipeline_project\\medium.pkl"

pickle.dump(cv, open(model_filepath, 'wb'))