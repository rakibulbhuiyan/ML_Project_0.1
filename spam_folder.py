import pandas as pd 
import matplotlib.pylab as plt

#_____________________ step 1 : import dataset

df = pd.read_csv("mail_data.csv")

# print(df.info())
# print(df.describe())


df['Category'] = df['Category'].map({'ham':1,'spam':0})
# print(df.head(5))

#______________________ step 3 : data processing

import re
import nltk
from nltk.corpus import stopwords

#  Download NLTK data
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer


lt = WordNetLemmatizer()
custom_words = set(stopwords.words('english')).union({'subject', 'regarding', 'please', 'thanks'})

def text_preprocessing(text):
    words = re.sub('[^a-zA-Z]',' ', text).lower().split()
    words = [ lt.lemmatize(word) for word in words if word not in custom_words]
    return ' '.join(words)

X = df['Message'].values     # the text data
y = df['Category'].values    # the labels: 1 for ham, 0 for spam


X_cleaned_data = [text_preprocessing(msg) for msg in X]
# print(X_cleaned_data[:5])

#_____________________ step 4 : Build , split and Train Data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

tf = TfidfVectorizer()
model = MultinomialNB()

X_vec = tf.fit_transform(X_cleaned_data)
X_train_vec,X_test,y_train,y_test = train_test_split(X_vec,y,test_size=.2,random_state=42)
model.fit(X_train_vec,y_train.ravel())


#_____________________ step 4 : Prediction and Evaluate data

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#_______________________ step : 5 spam vs ham

import seaborn as sns

sns.countplot(x='Category', data=df)
plt.xticks([0, 1], ['Spam', 'Ham'])
plt.title('Spam vs Ham Message Count')
plt.show()

#_______________________ step : 5 confusion metrics

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'], yticklabels=['Spam', 'Ham'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



