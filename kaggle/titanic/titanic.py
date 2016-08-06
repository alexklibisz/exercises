import sys
import pandas
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--clean', help='clean up the data', action='store_false')
ap.add_argument('-s', '--summary', help='show data summaries', action='store_true')
args = vars(ap.parse_args())

fname = './data-train.csv'
names = ['passengerid','survived','pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked']
df = pandas.read_csv(fname, names=names, skiprows=[0])

# Some cleaning up and recoding
if args['clean']:
    # remove passenger id, names, ticket
    df.drop(['passengerid','name','ticket'], axis=1, inplace=True)
    # recode categorical variables to integers using LabelEncoder
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'].astype('str'))
    le_cabin = LabelEncoder()
    df['cabin'] = le_cabin.fit_transform(df['cabin'].astype('str'))
    le_embarked = LabelEncoder()
    df['embarked'] = le_embarked.fit_transform(df['embarked'].astype('str'))

# High level summaries of the data
if args['summary']:
    print(df.head(10))
    print(df.describe())
    print(df.dtypes)
    print(df.groupby('survived').size())

    # Histogram to show distribution
    df.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
    plt.show()
    # Density plot to show distribution
    df.plot(kind='density', subplots=True, layout=(3,4), sharex=False, legend=True, fontsize=1)
    plt.show()
    # Box-and-whisker plot to show distribution
    df.plot(kind='box', subplots=True,layout=(3,4), sharex=False, sharey=False, fontsize=1, legend=True)
    plt.show()
