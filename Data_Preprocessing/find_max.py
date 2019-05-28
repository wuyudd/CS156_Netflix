# This file is to read in all the data and find max number of user, movie and date.

import pandas as pd
import numpy as np

# read in all the data and corresponding index
print("++++++++++++++++++ Start Loading all.dta ... ++++++++++++++++++")
df_all_with_date = pd.read_table('/Volumes/Disk1/ywwu/courses/cs156b/data/um/all.dta', delim_whitespace=True, header=None, names=['user', 'movie', 'date', 'rating'])
print("++++++++++++++++++ Now all.dta loaded successfully! ++++++++++++++++++")
df_all_date = df_all_with_date['date']
df_all_user = df_all_with_date['user']
df_all_movie = df_all_with_date['movie']
df_all_date_arr = df_all_date.values
df_all_user_arr = df_all_user.values
df_all_movie_arr = df_all_movie.values

max_date = np.max(df_all_date_arr)
max_user = np.max(df_all_user_arr)
max_movie = np.max(df_all_movie_arr)

print("max number of date = ", max_date)
print("max number of user = ", max_user)
print("max number of movie = ", max_movie)
