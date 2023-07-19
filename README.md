##  A small collection of some useful functions that come in handy from time to time (in progress).


Sometimes I write some code that might be part of a bigger project or some simple idea that comes to my mind and then I code them! There is no one single theme for this repo, but things that I found interesting and thought its good to keep track of them, or else


```
 "All those moments will be lost in time, like tears in rain."
 - Blade Runner, 1982 
```

---

#### Read Json file

```python

import json

# Open the JSON file in read mode
with open('file.json', 'rb') as file:

    # Load the contents of the file into a variable
    data = json.load(file

```


#### Save dictionary in a pickle file and read

```python

# Write

import pickle

# dictionary object to be saved
my_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

# open file in binary write mode
with open('my_dict.pkl', 'wb') as f:
    # dump dictionary object to file
    pickle.dump(my_dict, f)
    
    
# Read

import pickle
with open("name-of-file.pkl", "rb") as f:
    file = pickle.load(f)

print(file)
```

#### Write a dictionary object as json

```python
with open("file.json", "w") as f:
    json.dump(file, f, indent=4)
```

#### save a dataframe locally

```python

df.to_csv("FILE_NAME.csv", index=False)
````

#### Compare two dataframes and get the difference



```python
'''
Compare rows: You can compare the rows between two dataframes using the compare() method.
This method returns a dataframe containing the differences between two dataframes.

'''

import pandas as pd

df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [1, 2, 4], 'B': [4, 5, 6]})

diff = df1.compare(df2)
print(diff)
```


#### Creating data for cross validation 

```python

# Reference: https://stackoverflow.com/questions/61512087/use-kfolds-to-split-dataframe-i-want-the-rows-to-be-split-but-the-columns-are-g

import pandas as pd
from sklearn.model_selection import KFold

X = [[ 0.87, -1.34,  0.31, 1],
     [-2.79, -0.02, -0.85, 2],
     [-1.34, -0.48, -2.55, 3],
     [ 1.92,  1.48,  0.65, 4],
     [ 1.92,  1.48,  0.65, 5],
     [ 1.92,  1.48,  0.65, 6],
     [ 1.92,  1.48,  0.65, 7],
     [ 1.92,  1.48,  0.65, 8],
     [ 1.92,  1.48,  0.65, 9],
     [ 1.92,  1.48,  0.65, 10]]

finalDF = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'Target'])

print("=====Complete df======")
print(finalDF)

folds = KFold(n_splits=5)
scores = []
fold = 0

# you have to remove the target columsn by giving its column name.

for trainIndex, testIndex in folds.split(finalDF.drop(['Target'], axis=1)):
    fold += 1
    print(f"=========FOLD: {fold} starting ==============")
    xTrain = finalDF.loc[trainIndex, :]
    xTest = finalDF.loc[testIndex, :]
    print(xTrain.shape, xTest.shape)
    print(xTrain)
    print(xTest)

```

#### Splitting a df into train and test split only

:heavy_exclamation_mark: When you are saving the df using pickle then you need to have the same 
pandas version to unpickle it. For example I was saving a df in jupyterhub that is from the uni passau
and then I tried to unpickle it locally, it did not work because the pandas version were not same.

```python

'''
Splitting the arguments_df that has everything to train test split
'''
 
 
# Reference: https://stackoverflow.com/a/42932524/12946268

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Load the original DataFrame from CSV
df = pd.read_csv("COMPLETE_DATAFRAME.csv")


# Split into train (60%), validation (20%), and test (20%) sets
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)

with open(f"train_echr_42.pkl", 'wb') as f: pickle.dump(train_df, f)
with open(f"val_echr_42.pkl", 'wb') as f: pickle.dump(val_df, f)
with open(f"test_df_echr_42.pkl", 'wb') as f: pickle.dump(test_df, f)

'''
ATTENTION! DO NOT USE to_csv method from pandas it saves a dataframe to csv
file which is inconsistent with the original dataframe ( I have faced this issue, when
I was reading the csv file that was saved from to_csv method, I was getting null values
even tho in the original dataframe there was no null values)
'''

# To read

with open('test_df_echr_42.pkl', 'rb') as f:
    test = pickle.load(f)

'''
# Save the split DataFrames to CSV files

# DO NOT USE IT EVER
train_df.to_csv("train_echr_42.csv", index=False)
val_df.to_csv("validation_echr_42.csv", index=False)
test_df.to_csv("test_echr_42.csv", index=False)
'''
```


#### Get current directory

```python
import os
import datetime

def print_current_directory():
    current_directory = os.getcwd()
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current directory as of {time_now}:\n")
    print(f"{current_directory}")
print_current_directory()
```
