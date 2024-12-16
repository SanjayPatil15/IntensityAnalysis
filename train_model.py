import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load data
angriness_df = pd.read_csv('data/angriness.csv')
happiness_df = pd.read_csv('data/happiness.csv')
sadness_df = pd.read_csv('data/sadness.csv')
df = pd.concat([angriness_df, happiness_df, sadness_df], ignore_index=True)
df['intensity'] = df['intensity'].map({"anger": 0, "happiness": 1, "sadness": 2})  # Map labels to integers

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['content'], df['intensity'], test_size=0.2, random_state=42
)

# Tokenize data
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Convert to TensorFlow dataset
def create_dataset(encodings, labels):
    inputs = {key: tf.constant(val) for key, val in encodings.items()}
    dataset = tf.data.Dataset.from_tensor_slices((inputs, tf.constant(labels)))
    return dataset

train_dataset = create_dataset(train_encodings, train_labels.tolist()).shuffle(1000).batch(16)
test_dataset = create_dataset(test_encodings, test_labels.tolist()).batch(16)

# Load pre-trained model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Compile the model
optimizer = Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])

# Train the model
model.fit(train_dataset, epochs=3, validation_data=test_dataset)

# Save the model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate classification report
preds = model.predict(test_dataset)["logits"]
predicted_labels = tf.argmax(preds, axis=1)
print(classification_report(test_labels, predicted_labels.numpy(), target_names=["Anger", "Happiness", "Sadness"]))
