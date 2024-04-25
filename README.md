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

UPDATE
29.NOV.2023

The inconsistency was due to the nature of the dataset that has carraige return inside it '\r'
something like that. So if your dataset does not have any carraige return then you can
use to_csv happily
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


### Print in a nice way with datetime and current working directory

```python
def print_in_box(message: str) -> None:
    """
    Print a given message along with the current directory and timestamp in a box, separated by a horizontal line.

    Parameters:
    message (str): The message to be printed in the box.
    """
    # Get current directory and timestamp
    current_directory = os.getcwd()
    time_now = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

    # Prepare the directory and time information
    dir_info = f"Current directory as of {time_now}:\n{current_directory}"

    # Combine the custom message with the directory information, separated by a line
    combined_message = message + "\n\n" + "-" * len(max(message.split('\n'), key=len)) + "\n" + dir_info

    # Split the combined message into lines
    lines = combined_message.split('\n')
    # Find the length of the longest line
    max_length = max(len(line) for line in lines)
    # Create the top and bottom borders of the box
    top_border = "+" + "-" * (max_length + 2) + "+"
    bottom_border = top_border

    # Print the box with the combined message
    print(top_border)
    for line in lines:
        # Pad each line to the length of the longest line
        padded_line = line + ' ' * (max_length - len(line))
        print("| " + padded_line + " |")
    print(bottom_border)

```

### Print in a nice way but simpler version

```python
def print_in_box_simple(message: str) -> None:
    """
    Print a given message in a box.

    Parameters:
    message (str): The message to be printed in the box.
    """

    # Split the message into lines
    lines = message.split('\n')
    # Find the length of the longest line
    max_length = max(len(line) for line in lines)
    # Create the top and bottom borders of the box
    top_border = "+" + "-" * (max_length + 2) + "+"
    bottom_border = top_border

    # Print the box with the message
    print(top_border)
    for line in lines:
        # Pad each line to the length of the longest line
        padded_line = line + ' ' * (max_length - len(line))
        print("| " + padded_line + " |")
    print(bottom_border)

```

### Comparing two dataframe if they are identical or not

```python
def compare_dataframes(df1, df2):
    # Check if the shape of both dataframes is the same
    if df1.shape != df2.shape:
        return False, "Dataframes are not identical: Different shapes."

    # Check if columns and their order are the same
    if not df1.columns.equals(df2.columns):
        return False, "Dataframes are not identical: Different columns or column order."

    # Check if index and their order are the same
    if not df1.index.equals(df2.index):
        return False, "Dataframes are not identical: Different index or index order."

    # Check if the content of dataframes is the same
    if not df1.equals(df2):
        return False, "Dataframes are not identical: Different content."

    return True, "Dataframes are identical."

```

### Conversion to three decimal place with widgets

```python
import ipywidgets as widgets
from IPython.display import display

def round_to_three_decimals(number):
    rounded_number = round(number, 3)
    return rounded_number

def on_click(btn):
    number = float(input_text.value)
    output_label.value = f"{round_to_three_decimals(number)}"

input_text = widgets.FloatText(description="Number:")
submit_btn = widgets.Button(description="Convert")
submit_btn.on_click(on_click)
output_label = widgets.Label()

display(input_text, submit_btn, output_label)

```
### Save pandas dataframe in the current directory as csv and xlsx
```python
def save_dataframe(df, filename):
    """
    Saves the given DataFrame to both CSV and Excel formats in the current working directory.

    Args:
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The base filename without extension to use for saving the files.

    Returns:
    None
    """
    # Define file paths
    csv_file = f"{filename}.csv"
    excel_file = f"{filename}.xlsx"
    
    # Save as CSV
    df.to_csv(csv_file, index=False)
    print(f"DataFrame saved as CSV in {csv_file}")
    
    # Save as Excel
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"DataFrame saved as Excel in {excel_file}")

# Example usage
# Assuming df_data is your DataFrame and you want to save it as 'data'
save_dataframe(df_data, './data/after-clean/complete_data')
```
