
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

# # Looking around the data set

# In[154]:

get_ipython().magic(u'reset')


# In[155]:

# Import the required modules
import pandas as pd
import numpy as np
import scipy as sp


# In[156]:

# simple function to read in the user data file.
# the argument parse_dates takes in a list of colums, which are to be parsed as date format
user_data_raw_csv = pd.read_csv("/home/eyebell/local_bin/janacare/janCC/datasets/Habits-Data_upto-7th-May.csv",                            parse_dates = [-3, -2, -1])


# In[157]:

# import the pyexcel module
#import pyexcel as pe
#from pyexcel.ext import xls

# load the file
#records = pe.get_records(file_name="/home/eyebell/local_bin/janacare/datasets/Habits-Data_upto-7th-May.xls")
#len(records)
#for record in records:
    #print record


# In[158]:

# data metrics
user_data_raw_csv.shape # Rows , colums


# In[159]:

# data metrics
user_data_raw_csv.dtypes # data type of colums


# In[160]:

user_data_to_clean = user_data_raw_csv.copy()


# In[161]:

# Some basic statistical information on the data
#user_data_to_clean.describe()


# # Data Clean up

# In the last section of looking around, I saw that a lot of rows do not have any values or have garbage values(see first row of the table above).
# This can cause errors when computing anything using the values in these rows, hence a clean up is required.

# If a the coulums *last_activity* and *first_login* are empty then drop the corresponding row !

# In[162]:

# Lets check the health of the data set
user_data_to_clean.info()


# As is visible from the last column (*age_on_platform*) data type, Pandas is not recognising it as date type format. 
# This will make things difficult, so I delete this particular column and add a new one.
# Since the data in *age_on_platform* can be recreated by doing *age_on_platform* = *last_activity* - *first_login* 

# But on eyeballing I noticed some, cells of column *first_login* have greater value than corresponding cell of *last_activity*. These cells need to be swapped, since its not possible to have *first_login* > *last_activity*
# Finally the columns *first_login*, *last_activity* have missing values, as evident from above table. Since this is time data, that in my opinion should not be imputed, we will drop/delete the columns.

# In[163]:

# Run a loop through the data frame and check each row for this anamoly, if found drop,
# this is being done ONLY for selected columns

import datetime

swapped_count = 0
first_login_count = 0
last_activity_count = 0
email_count = 0
userid_count = 0

for index, row in user_data_to_clean.iterrows():        
        if row.last_activity == pd.NaT or row.last_activity != row.last_activity:
            last_activity_count = last_activity_count + 1
            #print row.last_activity
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


# In[164]:

user_data_to_clean.shape


# In[165]:

# Create new column 'age_on_platform' which has the corresponding value in date type format
user_data_to_clean["age_on_platform"] = user_data_to_clean["last_activity"] - user_data_to_clean["first_login"]


# In[166]:

user_data_to_clean.info()


# #### Validate if email i'd is correctly formatted and the email i'd really exists

# In[167]:

from validate_email import validate_email

email_count_invalid = 0
for index, row in user_data_to_clean.iterrows():        
        if not validate_email(row.email): # , verify=True)  for checking if email i'd actually exits
            user_data_to_clean.drop(index, inplace=True)
            email_count_invalid = email_count_invalid + 1
            
print "Number of email-id invalid: %d" % (email_count_invalid)


# In[168]:

# Check the result of last operation 
user_data_to_clean.info()


# ### Remove duplicates

# In[169]:

user_data_to_deDuplicate = user_data_to_clean.copy()


# In[170]:

user_data_deDuplicateD = user_data_to_deDuplicate.loc[~user_data_to_deDuplicate.email.str.strip().duplicated()]
len(user_data_deDuplicateD)


# In[171]:

user_data_deDuplicateD.info()


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


# # Clustering using Mean shift
# 
# from sklearn.cluster import MeanShift, estimate_bandwidth
# 
# #x = [1,1,5,6,1,5,10,22,23,23,50,51,51,52,100,112,130,500,512,600,12000,12230]
# x = pd.Series(user_data_deDuplicateD_timedelta64_converted['age_on_platform'])
# 
# X = np.array(zip(x,np.zeros(len(x))), dtype=np.int)
# '''--
# bandwidth = estimate_bandwidth(X, quantile=0.2)
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# 
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
# 
# for k in range(n_clusters_):
#     my_members = labels == k
#     print "cluster {0} : lenght = {1}".format(k, len(X[my_members, 0]))
#     #print "cluster {0}: {1}".format(k, X[my_members, 0])
#     cluster_sorted = sorted(X[my_members, 0])
#     print "cluster {0} : Max = {2} days & Min {1} days".format(k, cluster_sorted[0]*1.15741e-5, cluster_sorted[-1]*1.15741e-5)
# '''
# # The following bandwidth can be automatically detected using
# bandwidth = estimate_bandwidth(X, quantile=0.7)
# 
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(X)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
# 
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
# 
# print("number of estimated clusters : %d" % n_clusters_)
# for k in range(n_clusters_):
#     my_members = labels == k
#     print "cluster {0} : lenght = {1}".format(k, len(X[my_members, 0]))
#     cluster_sorted = sorted(X[my_members, 0])
#     print "cluster {0} : Min = {1} days & Max {2} days".format(k, cluster_sorted[0]*1.15741e-5, cluster_sorted[-1]*1.15741e-5)
# 
# # Plot result
# import matplotlib.pyplot as plt
# from itertools import cycle
# 
# %matplotlib inline
# 
# plt.figure(1)
# plt.clf()
# 
# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=14)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# 
# 

# In[174]:

# Clustering using Kmeans, not working
'''
y = [1,1,5,6,1,5,10,22,23,23,50,51,51,52,100,112,130,500,512,600,12000,12230]
y_float = map(float, y)
x = range(len(y))
x_float = map(float, x)

m = np.matrix([x_float, y_float]).transpose()


from scipy.cluster.vq import kmeans
kclust = kmeans(m, 5)

kclust[0][:, 0]

assigned_clusters = [abs(cluster_indices - e).argmin() for e in x]
'''


# ## Binning based on **age_on_platform** 
# day 1; day 2; week 1; week 2; week 3; week 4; week 6; week 8; week 12; 3 months; 6 months; 1 year; 

# In[175]:

user_data_binned = user_data_deDuplicateD_timedelta64_converted.copy()
                   
# function to convert age_on_platform in seconds to hours
convert_sec_to_hr = lambda x: x/3600
user_data_binned["age_on_platform"] = user_data_binned['age_on_platform'].map(convert_sec_to_hr).copy()

# filter rows based on first_login value after 30th April
user_data_binned_post30thApril = user_data_binned[user_data_binned.first_login < datetime.datetime(2016, 4, 30)]

for index, row in user_data_binned_post30thApril.iterrows():
    if row["age_on_platform"] < 25:
        user_data_binned_post30thApril.set_value(index, 'bin', 1)
        
    elif row["age_on_platform"] >= 25 and row["age_on_platform"] < 49:
        user_data_binned_post30thApril.set_value(index, 'bin', 2)    
        
    elif row["age_on_platform"] >= 49 and row["age_on_platform"] < 169: #168 hrs = 1 week
        user_data_binned_post30thApril.set_value(index, 'bin', 3)
        
    elif row["age_on_platform"] >=169 and row["age_on_platform"] < 337: # 336 hrs = 2 weeks
        user_data_binned_post30thApril.set_value(index, 'bin', 4)
        
    elif row["age_on_platform"] >=337 and row["age_on_platform"] < 505: # 504 hrs = 3 weeks
        user_data_binned_post30thApril.set_value(index, 'bin', 5)
        
    elif row["age_on_platform"] >=505 and row["age_on_platform"] < 673: # 672 hrs = 4 weeks
        user_data_binned_post30thApril.set_value(index, 'bin', 6)
        
    elif row["age_on_platform"] >=673 and row["age_on_platform"] < 1009: # 1008 hrs = 6 weeks
        user_data_binned_post30thApril.set_value(index, 'bin', 7)
        
    elif row["age_on_platform"] >=1009 and row["age_on_platform"] < 1345: # 1344 hrs = 8 weeks
        user_data_binned_post30thApril.set_value(index, 'bin', 8)
        
    elif row["age_on_platform"] >=1345 and row["age_on_platform"] < 2017: # 2016 hrs = 12 weeks
        user_data_binned_post30thApril.set_value(index, 'bin', 9)
        
    elif row["age_on_platform"] >=2017 and row["age_on_platform"] < 4381: # 4380 hrs = 6 months
        user_data_binned_post30thApril.set_value(index, 'bin', 10)
        
    elif row["age_on_platform"] >=4381 and row["age_on_platform"] < 8761: # 8760 hrs = 12 months
        user_data_binned_post30thApril.set_value(index, 'bin', 11)
        
    elif row["age_on_platform"] > 8761: # Rest, ie. beyond 1 year
        user_data_binned_post30thApril.set_value(index, 'bin', 12)
        
    else:
        user_data_binned_post30thApril.set_value(index, 'bin', 0)
    


# In[176]:

user_data_binned_post30thApril.info()


# In[177]:

print "Number of users with age_on_platform equal to 1 day or less, aka 0th day = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 1])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 1].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_0day.csv", index=False)


# In[178]:

print "Number of users with age_on_platform between 1st and 2nd days = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 2])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 2].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_1st-day.csv", index=False)


# In[179]:

print "Number of users with age_on_platform greater than or equal to 2 complete days and less than 1 week = %d" % len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 3])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 3].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_1st-week.csv", index=False)


# In[180]:

print "Number of users with age_on_platform between 2nd week = %d" % len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 4])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 4].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_2nd-week.csv", index=False)


# In[181]:

print "Number of users with age_on_platform between 3rd weeks = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 5])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 5].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_3rd-week.csv", index=False)


# In[182]:

print "Number of users with age_on_platform between 4th weeks = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 6])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 6].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_4th-week.csv", index=False)


# In[183]:

print "Number of users with age_on_platform greater than or equal to 4 weeks and less than 6 weeks = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 7])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 7].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_4th-to-6th-week.csv", index=False)


# In[184]:

print "Number of users with age_on_platform greater than or equal to 6 weeks and less than 8 weeks = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 8])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 8].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_6th-to-8th-week.csv", index=False)


# In[185]:

print "Number of users with age_on_platform greater than or equal to 8 weeks and less than 12 weeks = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 9])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 9].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_8th-to-12th-week.csv", index=False)


# In[186]:

print "Number of users with age_on_platform greater than or equal to 12 weeks and less than 6 months = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 10])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 10].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_12thweek-to-6thmonth.csv", index=False)


# In[187]:

print "Number of users with age_on_platform greater than or equal to 6 months and less than 1 year = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 11])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 11].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_6thmonth-to-1year.csv", index=False)


# In[188]:

print "Number of users with age_on_platform greater than 1 year = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 12])
user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 12].to_csv("/home/eyebell/local_bin/janacare/janCC/datasets/user_retention_email-campaign/user_data_binned_post30thApril_beyond-1year.csv", index=False)


# In[189]:

print "Number of users with age_on_platform is wierd = %d" %len(user_data_binned_post30thApril[user_data_binned_post30thApril.bin == 0])


# In[190]:

# Save dataframe with binned values as CSV
#user_data_binned_post30thApril.to_csv('user_data_binned_post30thApril.csv')


# In[ ]:




# 
# 
