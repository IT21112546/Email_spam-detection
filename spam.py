#Import libraries
import numpy as np 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import string

#Load the data and print the first 5 rows.*
#Load the data
#from google.colab import files # Use to load data on Google Colab
#uploaded = files.upload() # Use to load data on Google Colab
df = pd.read_csv('emails.csv')
df.head(5)

#Print the shape (Get the number of rows and cols)
df.shape


#Get the column names
df.columns

#Checking for duplicates and removing them
df.drop_duplicates(inplace = True)

*Show the new number of the rows and columns (if any) .

#Show the new shape (number of rows & columns)
df.shape

*Show the number of missing data for each column.

#Show the number of missing (NAN, NaN, na) data for each column
df.isnull().sum()

*Download the stop words. Stop words in natural language processing, are useless words (data).

#Need to download stopwords
nltk.download('stopwords')

*Create a function to clean the text and return the tokens. The cleaning of the text can be done by first removing punctuations and then removing the useless words also known as stop words.

#Tokenization (a list of tokens), will be used as the analyzer
#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    
    #1 Remove Punctuationa
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2 Remove Stop Words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3 Return a list of clean words
    return clean_words
	

#Show the Tokenization (a list of tokens )
df['text'].head().apply(process_text)

#Convert the text into a matrix of token counts.

from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

#Split data into 80% training & 20% testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)


#Get the shape of messages_bow
messages_bow.shape

#Create and train the Multinomial Naive Bayes classifier which is suitable for classification with discrete features (e.g., word counts for text classification)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


#Print the predictions
print(classifier.predict(X_train))
#Print the actual values
print(y_train.values)



#Evaluate the model on the training data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))

#Print the predictions
print('Predicted value: ',classifier.predict(X_test))
#Print Actual Label
print('Actual value: ',y_test.values)


#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))

