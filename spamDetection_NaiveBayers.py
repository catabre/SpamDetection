##Using Naive Bayes Algorithm to detect if a message is spam or not. Used training data set provided online####
##### 1: SPAM, 0: Not a SPAM #####

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("smsspamcollection/SMSSpamCollection", '\t')

#print(df.shape)
#print(df.head())
#print(df.values)
#print(df['message'])

d = {'ham':0, 'spam':1}

df['label'] = df.output.apply(lambda x:d[x])

#print(df.head())
#print(df.shape)

X_train, X_test, Y_train, Y_test = train_test_split(df['message'], df['label'], random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(Y_train.shape[0]))
print('Number of rows in the test set: {}'.format(Y_test.shape[0]))

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

#print(training_data.toarray())

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, Y_train)

predictions = naive_bayes.predict(testing_data)

#print("predictions")
#print(predictions)

print('Accuracy score: ', format(accuracy_score(Y_test, predictions)))
print('Precision score: ', format(precision_score(Y_test, predictions)))
print('Recall score: ', format(recall_score(Y_test, predictions)))
print('F1 score: ', format(f1_score(Y_test, predictions)))


##Custom message to predict whether it is spam or not. PLease try Yourself. Your own message####

#X_test = ["Apply for RBL Bank Platinum Maxima Credit Card. Get free movie ticket every month, lounge access, 20,000 bonus rewards & more."]
X_test = ["Our playground is all set to book and play. Register through http://smssd.in/sdadad or call us on 9875455627"]
testing_data = count_vector.transform(X_test)
predictions = naive_bayes.predict(testing_data)

print("Prediction for the Message: ")
print(predictions)

