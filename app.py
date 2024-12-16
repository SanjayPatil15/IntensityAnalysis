from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # Get prediction
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        # Map prediction to label
        labels = {0: "Anger", 1: "Happiness", 2: "Sadness"}
        result = labels[prediction]
        return render_template("index.html", text=text, result=result)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
