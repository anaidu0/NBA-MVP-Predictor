from basketball_reference_web_scraper import client
import pandas as pd

client.output_type = "OutputType.JSON"
advData = client.players_advanced_season_totals(season_end_year=2020)
seasonTotal = client.players_season_totals(season_end_year=2020)

json_file1 = pd.DataFrame(advData)
json_file2 = pd.DataFrame(seasonTotal)
json_file1.to_csv("adv_totals.csv")
json_file2.to_csv("total_totals.csv")