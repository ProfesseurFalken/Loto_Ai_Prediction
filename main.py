##################################################################
# !/usr/bin/env python
# -*- coding: utf-8 -*-
##################################################################
# Main_v2.py: Loto prediction Ai.
##################################################################
# __author__ = "Emmanuel Jean Louis Wojcik"
# __copyright__ = "Copyright 2024, The Joshua Project"
# __credits__ = ["Emmanuel Jean Louis Wojcik"]
##################################################################
# __license__ = "MIT"
# __version__ = "1.0.1"
# __maintainer__ = "Emmanuel Jean Louis Wojcik"
# __email__ = "wojcikej@orange.fr"
# __status__ = "Development"
##################################################################

##################################################################
# Import Libraries
##################################################################
from bs4 import BeautifulSoup
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, TimeDistributed, RepeatVector, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

##################################################################
# Scrap the loto numbers
##################################################################
def scrap_loto_numbers():
    # Initialize an empty list to store the lottery numbers
    my_list = []
    # Wait for 2 seconds before sending a request
    time.sleep(2)
    # URL of the website containing the lottery numbers
    loto_url = "http://loto.akroweb.fr/loto-historique-tirages/"
    # Send a GET request to the URL
    page = requests.get(loto_url)
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(page.text, 'html.parser')
    # Find the table containing the lottery numbers
    body = soup.find('table')
    # Find all rows in the table
    tirage_line = body.find_all('tr')
    # Loop through each row
    for value in tirage_line:
        my_dict = {}
        # Split the text content of the row by newline characters
        res = value.text.split('\n')
        # Check if the row contains enough values
        if len(res) < 11:  # Adding check for valid rows
            continue
        # Extract the day and month/year
        my_dict['day'] = res[2]
        my_dict['month_year'] = res[3]
        # Extract the lottery numbers
        for i, val in enumerate(res[5:10]):
            my_dict['num' + str(i)] = int(val)
        # Extract the chance number
        my_dict['chance'] = int(res[10])
        # Append the dictionary to the list
        my_list.append(my_dict)
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(my_list)
    return df

# Additional utility functions for feature engineering
def is_under(data, number):
    # Check if each number is less than or equal to the given number
    return ((data['num0'] <= number).astype(int) +
            (data['num1'] <= number).astype(int) +
            (data['num2'] <= number).astype(int) +
            (data['num3'] <= number).astype(int) +
            (data['num4'] <= number).astype(int))

def is_pair(data):
    # Check if each number is an even number
    return ((data['num0'].isin(pairs)).astype(int) +
            (data['num1'].isin(pairs)).astype(int) +
            (data['num2'].isin(pairs)).astype(int) +
            (data['num3'].isin(pairs)).astype(int) +
            (data['num4'].isin(pairs)).astype(int))

def is_impair(data):
    # Check if each number is an odd number
    return ((data['num0'].isin(impairs)).astype(int) +
            (data['num1'].isin(impairs)).astype(int) +
            (data['num2'].isin(impairs)).astype(int) +
            (data['num3'].isin(impairs)).astype(int) +
            (data['num4'].isin(impairs)).astype(int))

def is_pair_etoile(data):
    # Check if the chance number is an even number
    return (data['chance'].isin(pairs)).astype(int)

def is_impair_etoile(data):
    # Check if the chance number is an odd number
    return (data['chance'].isin(impairs)).astype(int)

def sum_diff(data):
    # Calculate the sum of the squared differences between consecutive numbers
    return ((data['num1'] - data['num0']) ** 2 +
            (data['num2'] - data['num1']) ** 2 +
            (data['num3'] - data['num2']) ** 2 +
            (data['num4'] - data['num3']) ** 2)

def freq_val(data, column):
    # Calculate the frequency of each number up to the current position
    tab = data[column].values.tolist()
    freqs = []
    pos = 1
    for e in tab:
        freqs.append(tab[0:pos].count(e))
        pos = pos + 1
    return freqs

# New feature engineering functions
def calculate_mean(data):
    # Calculate the mean of the lottery numbers
    return data[['num0', 'num1', 'num2', 'num3', 'num4']].mean(axis=1)

def calculate_median(data):
    # Calculate the median of the lottery numbers
    return data[['num0', 'num1', 'num2', 'num3', 'num4']].median(axis=1)

def calculate_std(data):
    # Calculate the standard deviation of the lottery numbers
    return data[['num0', 'num1', 'num2', 'num3', 'num4']].std(axis=1)

def calculate_range(data):
    # Calculate the range (max - min) of the lottery numbers
    return data[['num0', 'num1', 'num2', 'num3', 'num4']].max(axis=1) - data[
        ['num0', 'num1', 'num2', 'num3', 'num4']].min(axis=1)

def sum_numbers(data):
    # Calculate the sum of the lottery numbers
    return data[['num0', 'num1', 'num2', 'num3', 'num4']].sum(axis=1)

def odd_even_ratio(data):
    # Calculate the ratio of odd to even numbers
    odd_count = (data[['num0', 'num1', 'num2', 'num3', 'num4']] % 2).sum(axis=1)
    even_count = 5 - odd_count
    return odd_count / even_count

# Lists for pairs and impairs
pairs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
impairs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]

# Scrape the data
df_tirage = scrap_loto_numbers()
# Reverse the DataFrame to have the most recent data last
df = df_tirage.iloc[::-1]
# Select only the columns with lottery numbers and chance number
df = df[['num0', 'num1', 'num2', 'num3', 'num4', 'chance']]

# Apply feature engineering
df['freq_num0'] = freq_val(df, 'num0')
df['freq_num1'] = freq_val(df, 'num1')
df['freq_num2'] = freq_val(df, 'num2')
df['freq_num3'] = freq_val(df, 'num3')
df['freq_num4'] = freq_val(df, 'num4')
df['freq_chance'] = freq_val(df, 'chance')
df['sum_diff'] = sum_diff(df)
df['pair_chance'] = is_pair_etoile(df)
df['impair_chance'] = is_impair_etoile(df)
df['pair'] = is_pair(df)
df['impair'] = is_impair(df)
df['is_under_24'] = is_under(df, 24)
df['is_under_40'] = is_under(df, 40)
df['mean'] = calculate_mean(df)
df['median'] = calculate_median(df)
df['std'] = calculate_std(df)
df['range'] = calculate_range(df)
df['sum'] = sum_numbers(df)
df['odd_even_ratio'] = odd_even_ratio(df)

# Remove any infinite or excessively large values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Model parameters
nb_label_feature = 6
UNITS = 100
BATCHSIZE = 30
EPOCH = 1000
OPTIMIZER = 'adam'
LOSS = 'mae'
DROPOUT = 0.1
window_length = 12

# Define LSTM model
def define_model(number_of_features, nb_label_feature):
    model = Sequential()
    model.add(LSTM(UNITS, input_shape=(window_length, number_of_features), return_sequences=True))
    model.add(LSTM(UNITS, dropout=0.1, return_sequences=False))
    model.add(Dense(nb_label_feature))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['acc'])
    return model

# Create dataset for LSTM
def create_lstm_dataset(df, window_length, nb_label_feature):
    number_of_rows = df.shape[0]
    number_of_features = df.shape[1]
    # Standardize the dataset
    scaler = StandardScaler().fit(df.values)
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

    # Initialize arrays for training data and labels
    train = np.empty([number_of_rows - window_length, window_length, number_of_features], dtype=float)
    label = np.empty([number_of_rows - window_length, nb_label_feature], dtype=float)
    # Create the LSTM dataset
    for i in range(0, number_of_rows - window_length):
        train[i] = transformed_df.iloc[i:i + window_length, 0: number_of_features].values
        label[i] = transformed_df.iloc[i + window_length: i + window_length + 1, 0:nb_label_feature].values

    return train, label, scaler

# Assuming df is your preprocessed DataFrame
train, label, scaler1 = create_lstm_dataset(df, window_length, nb_label_feature)

# Check if a saved model exists
try:
    best_model = load_model('best_model.keras')
    print("Loaded existing model.")
except:
    best_model = define_model(train.shape[2], nb_label_feature)
    print("No existing model found. Created a new model.")

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model with adjusted EarlyStopping
history = best_model.fit(
    train,
    label,
    batch_size=30,
    epochs=1000,
    verbose=2,
    validation_split=0.2,  # Use a validation split to monitor val_loss
    callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=200), checkpoint_callback]
#   Consider improving patience and compare...
#   callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=400), checkpoint_callback]
)

# Plot training loss
plt.plot(history.history['loss'])
plt.legend(['train_loss'])
plt.show()

# Make predictions
last_twelve = df.tail(window_length)
scaled_to_predict = scaler1.transform(last_twelve.values)

scaled_predicted_output = best_model.predict(np.array([scaled_to_predict]))

# Create a placeholder for all features and fill it with predictions
placeholder = np.zeros((1, df.shape[1]))
placeholder[0, :6] = scaled_predicted_output

# Inverse transform the placeholder without feature names
original_scale_pred = scaler1.inverse_transform(placeholder)

# Print only the first 6 elements (predictions)
print(original_scale_pred[0, :6].astype(int))
