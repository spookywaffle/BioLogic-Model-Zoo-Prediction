# BioLogic-Model-Zoo-Prediction

We utilized a pre-existing algorithm that we genericized with GPTs to be applicable to a broad range of data that we couldn't see inside to begin with.

The code accomplishes the following:

1. Loads data from a CSV file named 'nasa_animal_testing_data.csv.'
2. Preprocesses the data by removing rows with missing values, splitting it into features (X) and labels (y), and further dividing it into training, validation, and test sets.
3. Scales the features using standardization (optional but common).
4. Defines a neural network model with three dense layers for a binary classification task.
5. Compiles the model with the Adam optimizer and binary cross-entropy loss function.
6. Trains the model on the training data with 10 epochs and a batch size of 32, using the validation set for monitoring.
7. Evaluates the trained model on the test set, printing the test loss and accuracy.
8. Scales new data using the same standardization as applied to the training data.
9. Uses the trained model to make predictions on the new data.
