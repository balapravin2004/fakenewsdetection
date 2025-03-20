from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, static_folder='template', template_folder='template')

# Load the saved model
with open("finished_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# Load the training data
df_train = pd.read_csv("news.csv")
x_train = df_train["text"]
y_train = df_train["label"]

# Fit the TF-IDF vectorizer with the training data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
vectorizer.fit(x_train)

# Transform the training data with the fitted vectorizer
x_train_vectorized = vectorizer.transform(x_train)

# Fit the model with the training data
model.fit(x_train_vectorized, y_train)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = request.form['news']
        try:
            # Transform the news article using the fitted vectorizer
            news_vectorized = vectorizer.transform([news])
            # Predict whether the news is true or false
            predict = model.predict(news_vectorized)
            # Map the prediction to human-readable format
            prediction_label = "True" if predict[0] == 'REAL' else "False"
        except Exception as error:
            # Handle any errors that occur during prediction
            print(error)
            prediction_label = "Error"
        # Return the prediction result to the HTML template
        return render_template("prediction.html", predict=f"News headline is -> {prediction_label}")
    return render_template("prediction.html")

if __name__ == '__main__':
    app.run()
