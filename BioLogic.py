import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

data = pd.read_csv('nasa_animal_testing_data.csv')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, num_features)))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=[Accuracy()])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

new_data = pd.read_csv('new_nasa_animal_testing_data.csv')
X_new = preprocess(new_data)
predictions = model.predict(X_new)
