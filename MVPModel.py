import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense

# satisfies unknown error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
current_data = os.path.join('data', 'final_2020_stats.csv')
current_stats_df = pd.read_csv(current_data)
current_stats_df.replace('', np.nan, inplace=True)
current_stats_df.dropna(inplace=True)
players = current_stats_df['Player']
current_stats_df = current_stats_df.drop(['Player'], axis=1)
final_df = current_stats_df.drop(['Year'], axis=1)

# This function creates the logistic regression model
# works best for final MVP prediction after the season ends
def lr_model(X, y):
    X = X.drop(['MVP'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    print(X.columns.values)
    print(model.coef_)
    # print("Logistic Regression Accuracy:", accuracy_score(y_test, predictions))  # prints the accuracy of the model
    return model

# creates a random forest classifier, seems to work better for printing out the top 5 candidates
# also works better for mid season data
def random_forest(X, y):
    X = X.drop(['MVP'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=100, max_depth=None)
    classifier = clf.fit(X_train, y_train)
    predictions = classifier.predict_proba(X_test)
    return clf


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
    model.add(Dense(1, activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(x, y, epochs=25, batch_size=20)  # runs through 25 epochs with batch sizes of 20
    scores = model.evaluate(x, y)
    model.save('MVP_Model.h5')  # saves the model
    print("\nSequential Model Accuracy: %.2f%%" % (scores[1] * 100))  # Print accuracy score of model
    return model


def ModelHelper(model, df, players):
    predictions = model.predict_proba(df)
    # print(stats.describe(predictions))
    pd.DataFrame(predictions).to_csv("probabilities.csv")
    top5 = pd.DataFrame(predictions).nlargest(5, 1)
    prediction = pd.DataFrame(top5, columns=['Not MVP', 'MVP Chance']).to_csv('prediction.csv')
    prediction_df = pd.read_csv('prediction.csv')
    for player in prediction_df.head().itertuples():
        player_name = players.iloc[player[1]]
        print(player_name)

# SequentialModel = sequential_model(X,y)
# LRModel = lr_model(X, y)
RandomForest = random_forest(X, y)
ModelHelper(RandomForest, final_df, players)
