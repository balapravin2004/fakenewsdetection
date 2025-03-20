import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__, static_folder='template', template_folder='template')

# Load training data with proper encoding
try:
    df_train = pd.read_csv("news.csv", encoding="ISO-8859-1")  # Alternative encoding
except UnicodeDecodeError:
    df_train = pd.read_csv("news.csv", encoding="utf-8", errors="replace")  # Handle errors

# Prepare training data
x_train = df_train["text"]
y_train = df_train["label"]

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_vectorized = vectorizer.fit_transform(x_train)

# Initialize and train the model
model = PassiveAggressiveClassifier()
model.fit(x_train_vectorized, y_train)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = request.form['news']
        try:
            # Transform and predict
            news_vectorized = vectorizer.transform([news])
            predict = model.predict(news_vectorized)
            prediction_label = "True" if predict[0] == 'REAL' else "False"
        except Exception as error:
            print(f"Error: {error}")
            prediction_label = "Error"
        
        return render_template("prediction.html", predict=f"News headline is -> {prediction_label}")
    
    return render_template("prediction.html")

if __name__ == '__main__':
    app.run(debug=True)
