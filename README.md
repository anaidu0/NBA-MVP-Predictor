# NBA-MVP-Predictor
This project predicts the National Basketball Association's MVP Award given a dataset from the regular season. The model was created by analyzing several features that are relevant to measuring a player's success on the court. The Data_Scraper file finds the list of MVPs and cleans up all the player data to fit it into the model. Data starting from 1985 was collected, however, the model was constructed using data from 2000 - 2017. 

# Details
In order to use this algorithm, replace the 'final_2018_stats.csv' file with your own stats dataset in the data folder. Then update the following line of code by replacing the file name.
```
# loads the current data to make the prediction for the year 2018
current_data = os.path.join('data','final_2018_stats.csv')
current_stats_df = pd.read_csv(current_data)
```
The model is contained in the file: MVPModel.py

# Prediction
The 2018 MVP was sucessfully predicted.

```
# Model can be changed by replacing the 'LRModel' with the 'SequentialModel'
find_MVP(LRModel, final_df, players)  # finds MVP 

The MVP from the inputted dataset is: James Harden
```

# Built With
* Python
* Keras
* Numpy
* scikit-learn
* Pandas
