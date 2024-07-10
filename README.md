# Loto_Ai_Prediction

This project aims to predict lottery numbers using various machine learning techniques, specifically Long Short-Term Memory (LSTM) networks. The code scrapes historical lottery data, processes it, and trains a machine learning model to predict future lottery numbers.
Table of Contents

    Installation
    Usage
    Features
    Model Details
    Results
    Contributing
    License

Installation
Prerequisites

Ensure you have Python 3.7+ installed. You will also need the following Python libraries:

    BeautifulSoup
    requests
    numpy
    pandas
    matplotlib
    scikit-learn
    tensorflow

You can install the required libraries using pip:

bash

pip install beautifulsoup4 requests numpy pandas matplotlib scikit-learn tensorflow

Usage

    Clone this repository:

bash

git clone https://github.com/ProfesseurFalken/loto_Ai_Prediction-ai.git
cd loto-prediction-ai

    Run the script:

bash

python Main_v2.py

The script will scrape the latest lottery data, process it, train the LSTM model, and output the predicted numbers.
Features

    Web Scraping: Scrapes historical lottery data from the web.
    Feature Engineering: Creates additional features from the raw lottery data to enhance model performance.
    LSTM Model: Uses an LSTM network to predict the next set of lottery numbers.
    Model Checkpointing: Saves the best model during training.
    Prediction: Outputs predicted lottery numbers.

Model Details
Data Preprocessing

    Scrapes data from loto.akroweb.fr.
    Processes the data to include various features such as frequency, mean, median, standard deviation, range, sum of numbers, odd/even ratio, and pair/impair indicators.

LSTM Network

    Layers: Two LSTM layers with dropout regularization.
    Optimizer: Adam.
    Loss Function: Mean Absolute Error (MAE).
    Early Stopping: Stops training when the validation loss does not improve for 200 epochs.
    Checkpointing: Saves the best model based on validation loss.

Training

    Splits the data into training and validation sets.
    Uses a batch size of 30 and trains for a maximum of 1000 epochs.

Prediction

    Uses the trained model to predict the next set of lottery numbers.
    Inverse transforms the scaled predictions to original values.

Results

The model predicts the next set of lottery numbers based on historical data. The prediction includes five main numbers and two luck numbers.
Contributing

Contributions are welcome! Please open an issue or submit a pull request.
License

This project is licensed under the MIT License. See the LICENSE file for details.
