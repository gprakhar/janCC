
# coding: utf-8

# Hello World!
# This notebook describes the effort filter out users to resurrect with Digital Marketing

# Clean up data
# 
# de-duplicate : based on email i'd
#                
# Partitioning the Data:
# two methods - 
# 
# A) cluster the data  and see how many clusters are there: used **MeanShift method**
# 
# 
# B) Bin the data based on *age_on_platform*
# 
# email capaign to Ressurrect users
# April 30th will be the cuttoff for the first_login value, for Binning

import pandas as pd
import numpy as np
import scipy as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time_format', '-t', help='Unit of time to be used for binning. \n eg. -t hours \n -t days', dafault='hours')
parser.add_argument('--first_login_filter', '-f', help='cutoff date for first_login value, only rows before this value are inculded for binning. format: YYYYMMDD \n eg. -f 19990101', default='')
parser.add_argument('--bin_max', '-M', help='upper bound value for age_on_platform.\n eg. -M 48', type=int, dafault=25)
parser.add_argument('--bin_min', '-m', help='lower bound value for age_on_platform.\n eg. -m 48', type=int, default=0)
parser.add_argument('--input_file', '-in', help='input file name. with atleast first_login and last_seen colums')

first_login_filter = args.first_login_filter
time_format = args.time_format
bin_max = args.bin_max
bin_min = args.bin_min
input_file = args.input_file

# simple function to read in the user data file.
# the argument parse_dates takes in a list of colums, which are to be parsed as date format
user_data_raw_csv = pd.read_csv(input_file, parse_dates = [-3, -2, -1])



# In[160]:

user_data_to_clean = user_data_raw_csv.copy()


import datetime

swapped_count = 0
first_login_count = 0
last_activity_count = 0
email_count = 0
userid_count = 0

for index, row in user_data_to_clean.iterrows():        
        if row.last_activity == pd.NaT or row.last_activity != row.last_activity:
            last_activity_count = last_activity_count + 1
            user_data_to_clean.drop(index, inplace=True)

        elif row.first_login > row.last_activity:
            user_data_to_clean.drop(index, inplace=True)
            swapped_count = swapped_count + 1

        elif row.first_login != row.first_login or row.first_login == pd.NaT:
            user_data_to_clean.drop(index, inplace=True)
            first_login_count = first_login_count + 1

        elif row.email != row.email: #or row.email == '' or row.email == ' ':
            user_data_to_clean.drop(index, inplace=True)
            email_count = email_count + 1

        elif row.user_id != row.user_id:
            user_data_to_clean.drop(index, inplace=True)
            userid_count = userid_count + 1

print "last_activity_count=%d\tswapped_count=%d\tfirst_login_count=%d\temail_count=%d\tuserid_count=%d" % (last_activity_count, swapped_count, first_login_count, email_count, userid_count)



# In[165]:

# Create new column 'age_on_platform' which has the corresponding value in date type format
user_data_to_clean["age_on_platform"] = user_data_to_clean["last_activity"] - user_data_to_clean["first_login"]



# #### Validate if email i'd is correctly formatted and the email i'd really exists

# In[167]:

from validate_email import validate_email

email_count_invalid = 0
for index, row in user_data_to_clean.iterrows():        
        if not validate_email(row.email): # , verify=True)  for checking if email i'd actually exits
            user_data_to_clean.drop(index, inplace=True)
            email_count_invalid = email_count_invalid + 1
            
print "Number of email-id invalid: %d" % (email_count_invalid)



# ### Remove duplicates

# In[169]:

user_data_to_deDuplicate = user_data_to_clean.copy()


# In[170]:

user_data_deDuplicateD = user_data_to_deDuplicate.loc[~user_data_to_deDuplicate.email.str.strip().duplicated()]
print "Number of unique rows :%d" % len(user_data_deDuplicateD)



# In[172]:

# Now its time to convert the timedelta64 data type column named age_on_platform to seconds
def convert_timedelta64_to_sec(td64):
    ts = (td64 / np.timedelta64(1, 's'))
    return ts

user_data_deDuplicateD_timedelta64_converted = user_data_deDuplicateD.copy()
temp_copy = user_data_deDuplicateD.copy()
user_data_deDuplicateD_timedelta64_converted.drop("age_on_platform", 1)
user_data_deDuplicateD_timedelta64_converted['age_on_platform'] = temp_copy['age_on_platform'].apply(convert_timedelta64_to_sec)


# In[173]:

user_data_deDuplicateD_timedelta64_converted.info()



# ## Binning based on **age_on_platform** 

user_data_binned = user_data_deDuplicateD_timedelta64_converted.copy()

if time_format == 'hours':
    # function to convert age_on_platform in seconds to hours
    convert_sec_to_hr = lambda x: x/3600
    user_data_binned["age_on_platform"] = user_data_binned['age_on_platform'].map(convert_sec_to_hr).copy()
elif time_format == 'days':
    convert_sec_to_days = lambda x: x*1.15741e-5
    user_data_binned["age_on_platform"] = user_data_binned['age_on_platform'].map(convert_sec_to_days).copy()

# filter rows based on first_login value
if not isnull(first_login_filter):
    user_data_binned_post30thApril = user_data_binned[user_data_binned.first_login < datetime.datetime\
        (int(first_login_filter[:4]),int(first_login_filter[4:6]),int(first_login_filter[6:8]))]
else:
    user_data_binned_post30thApril = user_data_binned.copy()

for index, row in user_data_binned_post30thApril.iterrows():
    if row["age_on_platform"] >= bin_min and row["age_on_platform"] < bin_max:
        user_data_binned_post30thApril.set_value(index, 'bin', 0)

print "Number of users with age_on_platform between %d and %d %s = %d" % (bin_min, bin_max, time_format,\
 len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 0])

user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 0].to_csv("user_retention_email-campaign_data_binned.csv", index=False)

