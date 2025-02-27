import pandas as pd
import numpy as np
import itertools
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv("news.csv")

labels = df.label

x_train, x_test, y_train, y_test = train_test_split(df["text"], labels, test_size=0.2, random_state=20)

vector = TfidfVectorizer(stop_words="english" , max_df = 0.7)

tf_train = vector.fit_transform(x_train)

tf_test = vector.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tf_train, y_train)

y_pred = pac.predict(tf_test)

score = accuracy_score(y_test, y_pred)

print(f"Accuracy : {round(score*100,2)}%")

confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])


# Create and train the models
pac = PassiveAggressiveClassifier()
# Train your model here if needed...

# Save the model to a file
filename = 'finished_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(pac, file)
