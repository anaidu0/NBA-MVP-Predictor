#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np


# In[2]:


adv_2020 = os.path.join('data', 'adv_totals.csv')
totals = os.path.join('data', 'total_totals.csv')


# In[3]:


adv_2020_df = pd.read_csv(adv_2020)
totals_2020_df = pd.read_csv(totals)


# In[4]:


edited_2020_df = pd.DataFrame()


# In[5]:


edited_2020_df['Year']= int(2020)
edited_2020_df['Player']= adv_2020_df['name']
edited_2020_df['Age']= adv_2020_df['age']
edited_2020_df['G']= adv_2020_df['games_played']
edited_2020_df['MP']= adv_2020_df['minutes_played']
edited_2020_df['PER']= adv_2020_df['player_efficiency_rating']
edited_2020_df['TS%']= adv_2020_df['true_shooting_percentage']
edited_2020_df['WS']= adv_2020_df['win_shares']
edited_2020_df['VORP']= adv_2020_df['value_over_replacement_player']
edited_2020_df.dtypes


# In[6]:


ppg = (((totals_2020_df['made_field_goals']-totals_2020_df['made_three_point_field_goals'])*2)+totals_2020_df['made_free_throws']+(totals_2020_df['made_three_point_field_goals']*3))/totals_2020_df['games_played']
ppg_series = pd.Series(ppg)
edited_2020_df['PPG'] = ppg_series.values

rpg = (totals_2020_df['defensive_rebounds']+totals_2020_df['offensive_rebounds'])/totals_2020_df['games_played']
rpg_series = pd.Series(rpg)
edited_2020_df['RPG'] = rpg_series.values

apg = totals_2020_df['assists']/totals_2020_df['games_played']
apg_series = pd.Series(apg)
edited_2020_df['APG'] = apg_series.values

spg = totals_2020_df['steals']/totals_2020_df['games_played']
spg_series = pd.Series(spg)
edited_2020_df['SPG'] = spg_series.values

bpg = totals_2020_df['blocks']/totals_2020_df['games_played']
bpg_series = pd.Series(bpg)
edited_2020_df['BPG'] = bpg_series.values


# In[7]:


fg = totals_2020_df['made_field_goals']/totals_2020_df['attempted_field_goals']
fg_series = pd.Series(fg)
edited_2020_df['FG%'] = fg_series.values

ft = totals_2020_df['made_free_throws']/totals_2020_df['attempted_free_throws']
ft_series = pd.Series(ft)
edited_2020_df['FT%'] = ft_series.values

rb = edited_2020_df['RPG'] * (edited_2020_df['G']/edited_2020_df['G'].max()) * 82
rb_series = pd.Series(rb)
edited_2020_df['TRB'] = rb_series.values

ast = edited_2020_df['APG']* (edited_2020_df['G']/edited_2020_df['G'].max()) * 82
ast_series = pd.Series(ast)
edited_2020_df['AST'] = ast_series.values

stl = edited_2020_df['SPG']* (edited_2020_df['G']/edited_2020_df['G'].max()) * 82
stl_series = pd.Series(stl)
edited_2020_df['STL'] = stl_series.values

blk = edited_2020_df['BPG']* (edited_2020_df['G']/edited_2020_df['G'].max()) * 82
blk_series = pd.Series(blk)
edited_2020_df['BLK'] = blk_series.values

pts = edited_2020_df['PPG'] * (edited_2020_df['G']/edited_2020_df['G'].max()) * 82
pts_series = pd.Series(pts)
edited_2020_df['PTS'] = pts_series.values


# In[8]:


vorp = edited_2020_df['VORP']* (82/totals_2020_df['games_played']) 
vorp_series = pd.Series(vorp)
edited_2020_df['VORP'] = vorp_series.values

ws = edited_2020_df['WS']* (82/totals_2020_df['games_played']) 
ws_series = pd.Series(ws)
edited_2020_df['WS'] = ws_series.values

minutes = edited_2020_df['MP'] * (82/totals_2020_df['games_played']) 
minutes_series = pd.Series(minutes)
edited_2020_df['MP'] = minutes_series.values

edited_2020_df.loc[edited_2020_df['G'] < edited_2020_df['G'].max()/2, ['RPG', 'SPG', 'APG', 'PPG', 'VORP', 'WS', 'MP', 'PER']] = 0

games = (edited_2020_df['G']/edited_2020_df['G'].max()) * 82
games_series = pd.Series(games)
edited_2020_df['G'] = games_series.values


edited_2020_df.replace(np.nan, 0, inplace=True)


# In[9]:


edited_2020_df.to_csv("./data/final_2020_stats.csv", index=False)




