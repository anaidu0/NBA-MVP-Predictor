# NBA-MVP-Predictor
This project predicts the National Basketball Association's MVP Award given a dataset from the regular season. The model was created by analyzing several features that are relevant to measuring a player's success on the court. The Data_Scraper file finds the list of MVPs and cleans up all the player data to fit it into the model. Data starting from 1985 was collected, however, the model was constructed using data from 2000 - 2017. 

# Details
In order to use this algorithm, replace the 'final_2018_stats.csv' file with your own stats dataset in the data folder. Then update the following line of code by replacing the file name.
```
In [44]:
# loads the current data to make the prediction for the year 2018
current_data = os.path.join('data','final_2018_stats.csv')
current_stats_df = pd.read_csv(current_data)
```
The model is contained in the file: Final MVP Predictor.ipynb

# Prediction
The 2018 MVP was sucessfully predicted.

```
In [85]:
find_MVP(prediction_df) # calls the method to print the predicted MVP

The MVP from the inputted dataset is: James Harden
```

# Built With
* Python
* scikit-learn
* Pandas
