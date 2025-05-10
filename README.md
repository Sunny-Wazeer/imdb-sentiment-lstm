# IMDB Sentiment Analysis with LSTM

This project applies a Long Short-Term Memory (LSTM) neural network to classify movie reviews from the IMDB dataset as positive or negative. The model is implemented and trained using TensorFlow and Keras in a Google Colab environment.



## Overview

- **Task**: Binary text classification (sentiment analysis)
- **Dataset**: IMDB Movie Reviews (50,000 reviews)
- **Model**: LSTM-based neural network
- **Tools**: Python, TensorFlow, Keras, Pandas, NumPy
- **Platform**: Google Colab (.ipynb notebook)


## Features

- End-to-end data preprocessing pipeline:
  - Lowercasing, HTML tag removal, punctuation cleaning
- Tokenization and sequence padding
- LSTM model architecture with dropout regularization
- Accuracy and loss tracking over training epochs
- Custom review prediction with trained model
- Model saving and loading using `.h5` and `.keras` formats

---

## Dataset

-[IMDB Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Format: CSV file with two columns:
  - review: text of the review
  - sentiment: positive or negative

## Model Architecture

```python
Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=230),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
Embedding Layer: Transforms words into dense vectors.

LSTM Layer: Captures sequence dependencies.

Dropout Layer: Reduces overfitting.

Output Layer: Binary classification using sigmoid.

## How to Run

1. Open the notebook in Google Colab.
2. Upload the IMDB Dataset.csv file.
3. Run all cells to:
 - Preprocess the data
 - Train the LSTM model
 - Evaluate accuracy
 - Predict sentiment on new reviews

## Example Prediction
review = ["This movie was a huge disappointment."]
Clean, tokenize, pad, and predict
prediction = model.predict(padded_input)
print("Prediction: Positive" if prediction > 0.5 else "Prediction: Negative")



## Results

Validation Accuracy: ~88%
Handles HTML tags and punctuation
Supports real-time sentiment prediction

## Output Files

model_lstm2.h5
model_lstm2.keras

## Author
**Author:** [Sunny Wazeer](https://github.com/Sunny-Wazeer)


