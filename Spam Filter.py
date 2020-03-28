#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:05:33 2018

@author: Mariana
"""

import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import binom_test
import re

# Import Data
df = pd.read_excel(r'C:\Users\Mariana\Documents\Maestría\UK\Universidad de Edimburgo\2. Web Analytics\Project\Final code Q2\ProjectQ2\Input/InputReviewsRaw.xlsx',encoding='utf8')

# Count Review per User
pivot = pd.pivot_table(df, index='user',values='no', aggfunc=np.size)
pivot.columns = ['no reviews']

#calculates the mean of all the reviews
average = df['rating'].mean()

#define the midpoint that will divide a good and a bad product (can be modified according to the distribution of the data)
mp=3

# initialize dummy iteration algorithm
avg_diff = 30000

while avg_diff != 0:

    #evaluate if the review disagrees with the overall mid point or not and
    #creates a dummy variable where 1 means that this specific review disagrees
    conditions = [
    ((df['rating'] < mp) & (average >= mp) | (df['rating'] >= mp) & (average < mp)),
    ((df['rating'] > mp) & (average >= mp) | (df['rating'] <= mp) & (average < mp))         
    ]
    choices = [1, 0]

    # add dummy variable to df
    df['disagree'] = np.select(conditions, choices)         

    # creates a column that counts number of reviews for each user disgareeing with overall mean, where user is the index
    pivot2 = pd.pivot_table(df, index='user',values='disagree', aggfunc=np.sum)

    #Merge pivot tables of unique users for columns: # of reviews and # of disagreeing reviews
    full_pivot = pd.concat([pivot, pivot2], axis=1, join_axes=[pivot.index])    

    #adds a column with the weight for each user´s rating
    full_pivot['weight'] = 1-(full_pivot['disagree']/full_pivot['no reviews'])    

    # resets the index 
    full_pivot2 = full_pivot.reset_index()

    #merge complete database with database of unique users and add weight, # reviews and #disagreeing reviews to complete database
    df_new = pd.merge(df, full_pivot2, on='user', how='left')         

    #calculate weighted rating for each review
    df_new['weighted rating'] = df_new['rating']*df_new['weight']

    #calculate weighted average of all reviews
    average_new = df_new['weighted rating'].mean()

    #define variable for iteration condition
    avg_diff = average - average_new
    average = average_new

# probability of review being spam using final #of disagreeing reviews
sum_dis = df_new['disagree_x'].sum()
probability = sum_dis/len(df_new)

# spamicity estimate for each reviewer

df_new['probabilities'] = 1- binom.cdf(df_new['disagree_y']-1, df_new['no reviews'], probability)
df_new.to_csv('checkdisagree.csv')


conditiontest = [df_new['disagree_y']>1 , df_new['disagree_y']==0]
    
choicestest = [binom_test(df_new['disagree_y'], df_new['no reviews'], df_new['probabilities'], alternative='greater'), 0]
df_new['p-value test'] = np.select(conditiontest, choicestest)

df_new['p-value test']= binom_test(df_new['disagree_y'], df_new['no reviews'], df_new['probabilities'], alternative='greater')


print(binom_test(0, 1, 1, alternative='less')/2)

print(binom_test(0, 2, 1, alternative='less')/2)

print(binom_test(1, 1, 0.292921331, alternative='less')/2)

print(binom_test(1, 1, 0.292921331, alternative='less')/2)

print(binom_test(1, 2, 0.500039755, alternative='less')/2)

print(binom_test(2, 2, 0.085802906, alternative='less')/2)

print(binom_test(3, 3, 0.025133501, alternative='less')/2)

print(binom_test(4, 4, 0.007362139, alternative='less')/2)

print(binom_test(5, 5, 0.002156527, alternative='less'))
##check condition for statistic approximates a N(0,1)
#if  (len(df)*probability*(1-probability)) >=9:
#    print('Condition complied')
#else:
#    print('Cannot perform test')
#
##test statistic for binomial test
#df_new['t_statistic']=((df_new['probabilities']-probability)/np.sqrt(probability*(1-probability)))*np.sqrt(len(df))
#
##N(0,1) two tails (to check if p=!spamicity dor each user)
#z = 1.96
#condition = [np.absolute(df_new['t_statistic']) > z, np.absolute(df_new['t_statistic']) < z]
#choice = [1, 0]
#
##1 for reject H0 (keep binomial probability) and 0 for accepting the probability in HO
#df_new['Test'] = np.select(condition, choice)

#create probability_spam: if reject H0 (df_new['Test']=1 then df_new['probabilities'] otherwise probability
conditions2 = [(df_new['Test'] == 0), (df_new['Test'] == 1)]
choices2 = [probability, df_new.probabilities]
df_new['probability_spam'] = np.select(conditions2, choices2)
df_new['spamicity'] = 1 - df_new['probability_spam']

# keep reviews that were not identified as spam
review_stam = df_new[df_new.spamicity < 0.7].iloc[:,0:7]

# We now check a set of reviews (reviews_to_be_checked) where spamicity > 0.7:
reviews_to_be_checked = df_new[df_new.spamicity > 0.7].iloc[:,0:7]

# Filter reviews that occur multiple times in the same thread, we consider those as invalid
pivot1 = reviews_to_be_checked.pivot_table(index='content', values='thread', aggfunc=lambda x: len(x.unique()))
pivot2 = reviews_to_be_checked.pivot_table(index='content', values='thread', aggfunc='count')

# concetinate, and filter invalid reviews through difference of count and distinct count per thread
pivot2.columns = ['thread2']
pivot = pd.concat([pivot1, pivot2], axis=1)
pivot['diff'] = pivot['thread'] - pivot['thread2']
filtered_reviews = pivot[(pivot['diff'] == 0)].reset_index().content.tolist()

# only keep one review of those that have been postet in more than one thread but not the same thread
keeplist = []
for contentd in filtered_reviews:
    checklist = reviews_to_be_checked[reviews_to_be_checked.content == contentd]
    reviews_to_be_checked = reviews_to_be_checked[reviews_to_be_checked.content != contentd]
    keeplist.append(checklist.head(1))

if len(keeplist) != 0:
    # get only one dataframe
    reviews_to_add_again = pd.concat(keeplist)
else:
    reviews_to_add_again = pd.DataFrame()
    
# Definition of function that calculates Levenshtein distance
def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

# function to filter reviews that were rephrased but are from the same user
# call with reviews_to_add_again
def filterReviewsByDistance(df):
    
    #retrieve a list with all the reviews (content) per user
    user_list = set(df['user'].tolist())
    
    test_list = []
    for user in user_list:
        testdf = df[df['user'] == user]
        testdf = testdf[['user','content']]
        test_list.append(testdf)
    
    ed_dist=[]
    us_list=[]
    
    #loop that will compare consecutive pair of reviews
    for user in range(len(test_list)):
        temp_list= test_list[user]['content'].tolist()
        us= test_list[user]['user'].tolist()
    
        for i in range(len(temp_list)-1):
            rev1=temp_list[i]
            rev1= re.sub(r"[\n\d\W]"," ", rev1)
            rev2=temp_list[(i+1)]
            rev2= re.sub(r"[\n\d\W]"," ", rev2)
            med=edit_distance(rev1,rev2)
            us_list.append(us[0])
            ed_dist.append(med)
    
    #output: data frame of edit distance of reviews per user
    df2=pd.DataFrame(us_list, columns=['user'])
    df2['med']=pd.DataFrame(ed_dist)
    
    #sum all edit distance of consecutive pair of reviews
    pivot_distance = pd.pivot_table(df2, index='user',values='med', aggfunc=np.sum)
    
    ###this second data frame is possible already in your process since here I´m just retrieving the number of reviews per user
    pivot_rev = pd.pivot_table(df, index='user',values='content', aggfunc=np.size)
    full_pivot = pd.concat([pivot_distance, pivot_rev], axis=1, join_axes=[pivot_distance.index])
    full_pivot2 = full_pivot.reset_index()
    df_new = pd.merge(df, full_pivot2, on='user', how='left')
    
    #add a new column that will tell how many reviews to keep from the possible spammer with no equal content in the reviews
    conditions = [(df_new['med'] < 920),(df_new['med'] > 920)]
    choices = [1, 0]
    df_new['reviews_keep'] = np.select(conditions, choices)
    
    df_final = df_new[df_new.reviews_keep == 1]
    df_final = df_final.iloc[:, :-3]
    
    return df_final

# Update reviews_to_be_checked dataframe with filtered reviews
reviews_to_be_checked = filterReviewsByDistance(reviews_to_add_again)
reviews_to_be_checked = reviews_to_be_checked.rename(columns=lambda x: x.replace('content_x', 'content'))

# finally, from dataframe reviews_to_be_checked, keep only one review per user
user_list = set(reviews_to_be_checked['user'].tolist())
review_by_user = []
for user in user_list:
    testdf = reviews_to_be_checked[reviews_to_be_checked['user'] == user]
    testdf = testdf.head(1)
    review_by_user.append(testdf)
reviews_to_add_again = pd.concat(review_by_user)

# final output
reviews_checked = pd.concat([review_stam, reviews_to_add_again])
# print output to excel
reviews_checked.to_excel('/Users/jakobkaiser/Desktop/UoE/Media and Web Analytics/ProjectQ2/Output/Output_SpamFilter.xlsx', encoding='utf-8')


