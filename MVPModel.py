import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense

# Loads the player data into a pandas dataframe
player_data = os.path.join('data', 'edited_stats.csv')
df = pd.read_csv(player_data)
# print(len(df))

# Removes all the NaN rows and columns
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)

# displays data only after 2000
df = df[df['Year'] >= 2000]
y = df['MVP']  # Stores MVP list in this dataframe

# the x dataframe only contains the statistics to create the model
X = df.drop(['Player'], axis=1)
X = X.drop(['Year'], axis=1)

# Formats the current data set to fit the model
current_data = os.path.join('data', 'final_2018_stats.csv')
current_stats_df = pd.read_csv(current_data)
current_stats_df.replace('', np.nan, inplace=True)
current_stats_df.dropna(inplace=True)
players = current_stats_df['Player']
current_stats_df = current_stats_df.drop(['Player'], axis=1)
final_df = current_stats_df.drop(['Year'], axis=1)


# This function creates the logistic regression model
def lr_model(X, y):
    X = X.drop(['MVP'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, predictions))  # prints the accuracy of the model
    return model


# This is a customized model using Keras
def sequential_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    np.random.seed(5)  # random seed to reproduce model
    dataset = X_train.values  # convert Pandas dataframe to numpy array
    # split into input and output variables
    x = dataset[:, 0:19]
    y = dataset[:, 19]
    # Builds model
    model = Sequential()
    model.add(Dense(25, input_dim=19, activation='relu'))  # input_dim stores the value 19 for features in the model
    model.add(Dense(19, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(x, y, epochs=25, batch_size=20)  # runs through 25 epochs with batch sizes of 20
    scores = model.evaluate(x, y)
    model.save('MVP_Model.h5')  # saves the model
    print("\nSequential Model Accuracy: %.2f%%" % (scores[1] * 100))  # Print accuracy score of model
    return model


# This function exports the MVP predictions to a CSV file and locates the MVP
def find_MVP(model, df, players):
    # Converts dataframe to numpy array if the model is not the LRModel
    if model != LRModel:
        data_array = df.values
        current_prediction = model.predict(data_array)
    else:
        current_prediction = model.predict(df)
    # exports predictions to csv file
    prediction = pd.DataFrame(current_prediction, columns=['MVP']).to_csv('prediction.csv')
    prediction_df = pd.read_csv('prediction.csv')
    # locates the MVP from the csv file
    location_MVP = prediction_df.loc[prediction_df['MVP'] == 1].index[0]
    player_name = players.iloc[location_MVP]
    print("The MVP from the inputted dataset is:", player_name.split('\\')[0])


SequentialModel = sequential_model(X,y)
LRModel = lr_model(X, y)
find_MVP(LRModel, final_df, players)  # finds MVP
