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
def compare_dataframes(df1, df2, columns):
    # Compare element-wise differences
    differences = df1[columns].compare(df2[columns])
    print("Differences:\n", differences)

    # Check data types
    print("\nData Types DF1:\n", df1[columns].dtypes)
    print("\nData Types DF2:\n", df2[columns].dtypes)

    # Check indices
    print("\nIndices DF1:\n", df1[columns].index)
    print("\nIndices DF2:\n", df2[columns].index)

    # Check individual columns for equality
    for col in columns:
        equal = df1[col].equals(df2[col])
        print(f"\nColumn {col} equality: {equal}")

# Call the function to compare DataFrames
compare_dataframes(temp_pv2, pv2_biased_transformed_ranks, ['mean', 'variance', 'std_dev'])

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
from datetime import datetime
def save_dataframe(df, filename):
    """
    Saves the given DataFrame to both CSV and Excel formats in the specified directory.

    Args:
    df (pandas.DataFrame): The DataFrame to save.
    filename (str): The base filename without extension to use for saving the files.

    Returns:
    None
    """
    # Get the directory from the filename
    directory = os.path.dirname(filename)
    
    # Check if the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")
    
    # Define file paths
    csv_file = f"{filename}.csv"
    excel_file = f"{filename}.xlsx"
    
    # Save as CSV
    df.to_csv(csv_file, index=False)
    print(f"DataFrame saved as CSV in {csv_file}")
    
    # Save as Excel
    df.to_excel(excel_file, index=False, engine='openpyxl')
    print(f"DataFrame saved as Excel in {excel_file}")
    time_now = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
    print(f"Saved on {time_now}:\n")

# Example usage
# Assuming df_data is your DataFrame and you want to save it as 'data'
save_dataframe(df_data, './data/after-clean/complete_data')
```


### toggle pandas display settings
```python
def toggle_pandas_display_settings(mode='full'):
    """
    Toggle the display settings of pandas DataFrame.

    Parameters:
    - mode: 'full' to display DataFrames without truncation, 'default' to reset to default settings.
    
    Example:
    
    # To turn on full display:
    toggle_pandas_display_settings('full')

    # To reset to default settings:
    toggle_pandas_display_settings('default')

    """
    if mode == 'full':
        # Set to display DataFrames without truncation
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)  # For pandas versions 1.0 and later
        # pd.set_option('display.max_colwidth', -1)  # Uncomment for pandas versions before 1.0
        print("Pandas display settings set to full display mode.")
    elif mode == 'default':
        # Reset to pandas default display settings
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.max_colwidth')
        print("Pandas display settings reset to default.")
    else:
        print("Invalid mode. Please choose 'full' or 'default'.")
```

### Reads specified sheets from an Excel file using pandas.
```python
def read_excel_sheets(file_path, sheets=None, return_type='single'):
    """
    Reads specified sheets from an Excel file using pandas.

    :param file_path: str, path to the Excel file.
    :param sheets: str, int, or list, names or indices of the sheets to read.
    :param return_type: str, 'single' to return a single DataFrame (if one sheet is specified),
                        'dict' to return a dictionary of DataFrames (if multiple sheets are specified).
    :return: DataFrame or dict of DataFrames depending on return_type and sheets.
    """
    # Read the sheets based on the provided 'sheets' argument
    time_now = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
    try:
        data = pd.read_excel(file_path, sheet_name=sheets)
        print(f"{file_path} was read on {time_now}:\n")
    except Exception as e:
        print(f"Failed to read the file: {e}")
        return None

    # If multiple sheets are read into a dictionary
    if isinstance(data, dict):
        if return_type == 'single':
            # If user wants a single DataFrame but multiple sheets were requested, raise an error
            raise ValueError("Multiple sheets found but 'single' DataFrame requested. Specify correct 'return_type'.")
        return data
    else:
        if return_type == 'dict':
            # If user expects a dictionary but only one sheet was read, adjust the return structure
            print(f"{file_path} was read on {time_now}:\n")
            return {sheets: data}
        return data
# Example usage
data = read_excel_sheets(file_path='Data_complete_Can_GPT_Replace_Human_Examiners.xlsx',
                         sheets='Robustness & Extensions')
data.head(6)

```


### save dictionary as json
```python
def save_dict_as_json(d, filename):
    """
    Saves a dictionary as a JSON file, but only if the file does not already exist.

    Parameters:
    d (dict): The dictionary to save.
    filename (str): The path and name of the file to save the dictionary to.

    Raises:
    FileExistsError: If a file with the specified name alre exists.
    """

    # Check if the file already exists
    if os.path.exists(filename):
        raise FileExistsError(f"File '{filename}' already exists.")

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the dictionary as a JSON file
    with open(filename, "w") as file:
        json.dump(d, file, indent=4)

    # print_in_box(f"Result saved successfully at\n{filename}")
```


### Read excel sheets

```python
def read_excel_sheets(file_path, sheets=None, return_type="single"):
    """
    Reads specified sheets from an Excel file using pandas.

    :param file_path: str, path to the Excel file.
    :param sheets: str, int, or list, names or indices of the sheets to read.
    :param return_type: str, 'single' to return a single DataFrame (if one sheet is specified),
                        'dict' to return a dictionary of DataFrames (if multiple sheets are specified).
    :return: DataFrame or dict of DataFrames depending on return_type and sheets.
    """
    # Read the sheets based on the provided 'sheets' argument
    try:
        data = pd.read_excel(file_path, sheet_name=sheets)
    except Exception as e:
        print(f"Failed to read the file: {e}")
        return None

    # If multiple sheets are read into a dictionary
    if isinstance(data, dict):
        if return_type == "single":
            # If user wants a single DataFrame but multiple sheets were requested, raise an error
            raise ValueError(
                "Multiple sheets found but 'single' DataFrame requested. Specify correct 'return_type'."
            )
        return data
    else:
        if return_type == "dict":
            # If user expects a dictionary but only one sheet was read, adjust the return structure
            return {sheets: data}
        return data
```
### drop columns from padnas

```python
def drop_columns_from(df, start_column):
    """
    Drop all columns from the specified start_column to the end of the DataFrame (inclusive).

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.
    start_column (str): The column name from which to start dropping.

    Returns:
    pd.DataFrame: A DataFrame with the specified columns removed.
    """
    # Get the index of the start column
    start_index = df.columns.get_loc(start_column)

    # Get the column names to drop from start_index to the end
    columns_to_drop = df.columns[start_index:]

    # Drop the columns
    df = df.drop(columns=columns_to_drop)
    
    return df
```

### read csv/excel with fallback

```python
def read_file_with_fallback(primary_file, fallback_file):
    """
    Reads a file (CSV or Excel) into a DataFrame. If the primary file does not exist, it reads the fallback file.

    Parameters:
    primary_file (str): The path to the primary file (CSV or Excel).
    fallback_file (str): The path to the fallback file (CSV or Excel).

    Returns:
    pandas.DataFrame: DataFrame created from the read file.
    """
    # Determine which file to read
    file_to_read = primary_file if os.path.exists(primary_file) else fallback_file
    
    # Print message indicating which file is being read
    print(f"Reading from {'primary' if file_to_read == primary_file else 'fallback'} file: {file_to_read}")

    # Check file extension and read accordingly
    if file_to_read.endswith('.csv'):
        df = pd.read_csv(file_to_read)
    elif file_to_read.endswith('.xlsx'):
        df = pd.read_excel(file_to_read)
    else:
        raise ValueError("Unsupported file format. Only .csv and .xlsx are supported.")

    return df
```

### Convert to three decimal place

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
****
```

### select gpu device

```python
def select_gpu_device(device_id):

    """
    Note Please: Will be used when I am using huggingface parser and all the code in a script
    
    Selects a GPU device if available and prints information about all available GPUs.

    Args:
    device_id (int): The ID of the GPU device to use.

    Returns:
    str: The selected device, either a specific GPU or the CPU if no GPU is available.
    """
    # Check available GPUs and print their names
    
    gpu_count = torch.cuda.device_count()
    '''
    print("Available GPUs:", gpu_count)
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    '''
    # Choose a specific GPU based on device_id or fallback to CPU if GPUs are unavailable
    device = f"cuda:{device_id}" if torch.cuda.is_available() and device_id < gpu_count else "cpu"
    # print_in_box(f"Using device: {device}")
    
    return device
```

### Check for empty list and repeated elements in a list of a specified column of a pandas DataFrame

```python
def find_repeated_and_empty_elements(df, column_name):
    """
    This function checks each row in a specified column of a DataFrame for two conditions:
    1. If the list is empty.
    2. If there are repeated elements in the list.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be checked.
    column_name (str): The name of the column containing the string representations of lists.
    
    Outputs:
    Prints the row index and details if an empty list or repeated elements are found.
    """
    def get_repeated_elements(lst):
        """
        Helper function to find repeated elements in a list.
        
        Parameters:
        lst (list): The list to check for repeated elements.
        
        Returns:
        list: A list of elements that are repeated.
        """
        counter = Counter(lst)
        return [item for item, count in counter.items() if count > 1]
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Convert the string representation of the list to an actual list
        score = ast.literal_eval(row[column_name])
        
        # Check for empty list
        if not score:
            print(f"Row {index} has an empty list.")
        
        # Check for repeated elements
        repeated_elements = get_repeated_elements(score)
        if repeated_elements:
            print(f"Row {index} has repeated elements: {repeated_elements} in list {score}")

# Call the function with the DataFrame and the column name
find_repeated_and_empty_elements(df=df_llama3_rank_data,
                                 column_name='prompt_v1_rank_assessment_llama3_model_ranks')


```


### Check if two column from two different dataframe are identical or not

```python

def are_columns_identical(df1, col1, df2, col2):
    """
    Check if two columns from two different DataFrames are identical.
    
    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    col1 (str): The column name from the first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    col2 (str): The column name from the second DataFrame.
    
    Returns:
    bool: True if the columns are identical, False otherwise.
    """
    # Check if the columns exist in their respective DataFrames
    if col1 not in df1.columns or col2 not in df2.columns:
        raise ValueError(f"Column not found in DataFrame: {col1} in df1 or {col2} in df2")
    
    # Check if the lengths of the columns are the same
    if len(df1[col1]) != len(df2[col2]):
        return False
    
    # Check if all elements in the columns are the same
    return df1[col1].equals(df2[col2])
```

### Check if n number of columns from two different dataframe are identical or not

> The order must be the same as it is given to cols1 and cols2

```python
def are_multiple_columns_identical(df1, cols1, df2, cols2):
    """
    Check if multiple columns from two different DataFrames are identical.
    
    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    cols1 (list of str): The column names from the first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    cols2 (list of str): The column names from the second DataFrame.
    
    Returns:
    bool: True if all specified columns are identical, False otherwise.
    """
    # Check if the lengths of the column lists are the same
    if len(cols1) != len(cols2):
        raise ValueError("The number of columns to compare must be the same.")
    
    # Iterate through each pair of columns and check for equality
    for col1, col2 in zip(cols1, cols2):
        # Check if the columns exist in their respective DataFrames
        if col1 not in df1.columns or col2 not in df2.columns:
            raise ValueError(f"Column not found in DataFrame: {col1} in df1 or {col2} in df2")
        
        # Check if the lengths of the columns are the same
        if len(df1[col1]) != len(df2[col2]):
            return False
        
        # Check if all elements in the columns are the same
        if not df1[col1].equals(df2[col2]):
            return False
    
    return True
```

### ASCII-Banner

```bash

>>=======================<<
||                       ||
||                       ||
||                       ||
||   _  _   _     ___ _  ||
||  /  |_) |_  /\  | |_  ||
||  \_ | \ |_ /--\ | |_  ||
||                       ||
||                       ||
||                       ||
>>=======================<<

>>===================================================<<
|| Developer: Abdullah Al Zubaer                     ||   
|| Email: abdullahal.zubaer@uni-passau.de            ||
|| Institution: University of Passau                 ||
|| Project Page: https://www.uni-passau.de/deepwrite ||   
>>===================================================<<

```
### Return a subset of the df with specified column names

```python

def subset_dataframe(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Returns a subset of the DataFrame with only the specified columns.

    Parameters:
    df (pd.DataFrame): The original DataFrame.
    columns (list): A list of column names to include in the subset DataFrame.

    Returns:
    pd.DataFrame: A subset DataFrame containing only the specified columns, copied to ensure independence from the original DataFrame.
    Example:
    
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    
    # List of columns to subset
    columns_to_keep = ['A', 'C']
    
    # Get the subset DataFrame
    subset_df = subset_dataframe(df, columns_to_keep)
    """
    if not all(column in df.columns for column in columns):
        raise ValueError("One or more columns not found in the DataFrame")
    
    return df[columns].copy()
```
#### Filter Dataframe 

```python
import pandas as pd

def filter_dataframe(df, column_name, match_value, selected_columns=None):
    """
    Filters the DataFrame based on the given column name and matching value.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    column_name (str): The name of the column to filter by.
    match_value (str or int): The value to match in the specified column.
    selected_columns (list of str, optional): A list of column names to select after filtering. 
                                              If None, all columns are returned.
    
    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Filter the DataFrame
    filtered_df = df[df[column_name] == match_value]
    
    # If selected_columns is provided, select those columns
    if selected_columns is not None:
        filtered_df = filtered_df[selected_columns]
    
    return filtered_df

# Example usage:
# Assuming df is your DataFrame
# filtered_df = filter_dataframe(df, 'Task', 1)
# or with specific columns:
# filtered_df = filter_dataframe(df, 'Task', 1, ['Column1', 'Column2'])
# This will filter the DataFrame where the "Status" column equals "Completed" and return only the "Task" and "Assignee" columns.

```
#### Find and prinot NaN Rows in pandas dataframe

```python

def find_and_print_nan_rows(dataframe, column_name):
    if column_name not in dataframe.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return
    
    nan_rows = dataframe[dataframe[column_name].isna()]
    
    if nan_rows.empty:
        print(f"No NaN values found in column '{column_name}'.")
    else:
        print(f"Indices of rows with NaN values in column '{column_name}':")
        print(nan_rows.index.tolist())
        
# find_and_print_nan_rows(data, 'Answer')
```

#### Insert new line after every n words

```python
def insert_newline_every_n_words(text, n=12):
    # Split the text into words
    words = text.split()
    
    # Create a list to store lines
    lines = []
    
    # Loop over the words in chunks of size n
    for i in range(0, len(words), n):
        # Join the words in the chunk into a line and append to lines list
        lines.append(' '.join(words[i:i + n]))
    
    # Join all the lines with newlines
    result = '\n'.join(lines)
    
    return result

## Example usage:
# text = "[INSERT_YOUR_LONG TEXT]"
#formatted_text = insert_newline_every_n_words(text, 11)
#print(formatted_text)
```
#### Reorder words randomly from a string and return

```python
def reorder_words_randomly(text: str, n: int) -> str :
    
    """
    Reorder n words in the given text randomly and return the modified sentence.
    
    Args:
        text (str): The input sentence to be modified.
        n (int): The number of words to shuffle.
    
    Returns:
        str: The sentence with n words shuffled.
    """

    # Split the text into individual words
    words = text.split()
    total_words = len(words)
    
    # If n is greater than total words in the text, set n to total words
    n = min(n, total_words)
    
    # If n is 0 or the total words are less than 2, return the original text
    if n == 0 or total_words < 2:
        return text
    
    # Keep shuffling until the new order is different from the original
    while True:
        # Extract n random indices without replacement
        indices = random.sample(range(total_words), n)
        
        # Extract the words at those indices
        words_to_shuffle = [words[i] for i in indices]
        
        # Shuffle the extracted words
        random.shuffle(words_to_shuffle)
        
        # Put the shuffled words back into their original positions
        shuffled_words = words[:]
        for idx, word_idx in enumerate(indices):
            shuffled_words[word_idx] = words_to_shuffle[idx]
        
        # If the shuffled words are different from the original, return them
        if shuffled_words != words:
            return ' '.join(shuffled_words)
'''
# Example usage
text = "one two three four five six seven eight nine ten"
i=0
while i<10:
    i+=1
    shuffled_sentence = reorder_words_randomly(text, 2)
    print(shuffled_sentence)
'''
```

##### Compare the values of multiple columns from two DataFrames, ignoring column names

```python
def compare_columns_by_values(df1, cols1: list[str], df2, cols2: list[str], reset_index=True) -> bool:
    """
    Compare the values of multiple columns from two DataFrames, ignoring column names.
    The order of the columns must be the same in both DataFrames i.e.
    cols1[0] is compared to cols2[0], cols1[1] is compared to cols2[1], and so on.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        cols1 (list[str]): Columns to compare from the first DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        cols2 (list[str]): Columns to compare from the second DataFrame.
        reset_index (bool): Whether to reset the index before comparison. Default is True.

    Returns:
        bool: True if the selected columns are equal in both DataFrames, False otherwise.

    Example:
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df2 = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

    result = compare_columns_by_values(df1, ['a', 'b'], df2, ['x', 'y'])
    print(result)  # True

    # Order of the columns matters and it returns False
    result = compare_columns_by_values(df1, ['b', 'a'], df2, ['x', 'y'])
    print(result)  # False
    """
    # Extract subsets of the DataFrames
    subset1 = df1[cols1]
    subset2 = df2[cols2]
    
    # Optionally reset the index
    if reset_index:
        subset1 = subset1.reset_index(drop=True)
        subset2 = subset2.reset_index(drop=True)
    
    # Compare values
    return (subset1.values == subset2.values).all() and subset1.to_numpy().tolist() == subset2.to_numpy().tolist()

```


#### Check if the extraction function is working properly for columns contaiing the LLMs output

```python
'''
If there are errors then the regular expression needs modification

You need to know which columns you want to check the function with - then you pass the columns as a list of strings
'''


def extract_score(value) -> float:
    """Extracts the score from a string value formatted as 'Punktzahl: <number>'. Returns None if no score is found."""

    match = re.search(r'Punktzahl:\s*(\d+(?:\.\d+)?)', str(value))

    return float(match.group(1)) if match else 'ERROR_COULD_NOT_EXTRACT'

def apply_extract_score_and_report(df, columns_to_check, extract_score_func):
    """
    Applies the extract_score function to specified columns in the dataframe and tracks errors.
    Generates a summary report of the results.

    Parameters:
    - df (pandas.DataFrame): The dataframe containing the data.
    - columns_to_check (list of str): The list of column names to process.
    - extract_score_func (callable): The function used to extract the score.

    Returns:
    - dict: A summary report containing total errors and details per column.
    - int: Total number of errors found across all columns.
    """
    total_errors = 0
    summary_report = {}

    for col in columns_to_check:
        if col not in df.columns:
            print(f"NOT FOUND: Column '{col}' not found in dataframe. Skipping...")
            summary_report[col] = {
                'errors_found': False,
                'error_count': 0,
                'error_rows': [],
                'status': 'Column not found'
            }
            continue

        print(f"Processing column: {col}")
        extracted_col_name = f"{col}_Extracted"

        # Apply the function
        df[extracted_col_name] = df[col].apply(lambda x: extract_score_func(x))

        # Identify problematic rows
        errors = df[df[extracted_col_name] == 'ERROR_COULD_NOT_EXTRACT']
        error_count = len(errors)
        total_errors += error_count

        # Store error details in the summary report
        summary_report[col] = {
            'errors_found': error_count > 0,
            'error_count': error_count,
            'error_rows': errors.index.tolist(),
            'status': 'Processed with errors' if error_count > 0 else 'Processed successfully'
        }

        # Uncomment the following lines if you want to see the errors
        # if error_count > 0:
        #     print(f"\nIssues found in column '{col}':")
        #     for idx, row in errors.iterrows():
        #         print(f"Row {idx}: Text - {row[col]}")

    if total_errors == 0:
        print("\nNo errors found in any column.")
    else:
        print("\nErrors were found. See details above.")
        print(f"Total errors found: {total_errors}")

    # Return the summary report (empty or not)
    return summary_report, total_errors

#CHANGEHERE#
models = ["gemma2", "llama3.1", 'llama3', 'mistral']

#CHANGEHERE#
columns_to_check = [
    f"prompt_v1_original_macro_eco_de_complete_response_{model}_iteration_{i}"
    for model in models for i in range(1, 11)
]

# Call the function
summary_report, total_errors = apply_extract_score_and_report(df=df_to_score,
                                                              columns_to_check=columns_to_check,
                                                              extract_score_func=extract_score)

# Print the summary report only if errors were found
if total_errors > 0:
    print("\nSummary Report:")
    for col, details in summary_report.items():
        if details['errors_found']:
            print(f"Column: {col}")
            print(f"  Status: {details['status']}")
            print(f"  Errors Found: {details['errors_found']}")
            print(f"  Error Count: {details['error_count']}")
            print(f"  Error Rows: {details['error_rows']}")
else:
    print("\nNo errors to report.")

```
#### Filtering pandas dataframe based on specified string present in the df
```python
import pandas as pd

# Example dataframe
data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "score_math": [90, 85, 80],
    "score_english": [88, 92, 78],
    "total_score": [178, 177, 158],
}

df = pd.DataFrame(data)

# Use 'filter' with 'regex' to select columns matching the regex pattern
regex_pattern = r"score"  # Matches any column containing "score"
filtered_df = df.filter(regex=regex_pattern, axis=1)

print(filtered_df)
```

####  Compare two strings using SequenceMatcher and print the differences.
```python
from difflib import SequenceMatcher

def compare_strings(str1, str2):
    """
    Compare two strings using SequenceMatcher and print the differences.

    Args:
        str1 (str): The first string to compare.
        str2 (str): The second string to compare.
    """
    matcher = SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        print(f"{tag}: '{str1[i1:i2]}' vs '{str2[j1:j2]}'")

# Example usage:
# compare_strings("string1", "string2")
```
#### Check any string if present in any column in DF
```python
def check_string_in_any_column(df, string_to_check):
    """
    Check if a given string is present in any column name of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    string_to_check (str): The substring to look for in column names.

    Returns:
    bool: True if the string is present in any column name, False otherwise.
    """
    # Filter and print columns containing the substring
    matching_columns = [col for col in df.columns if string_to_check in col]
    
    if matching_columns:
        print(f"Columns containing '{string_to_check}': {matching_columns}")
        print(f"Total columns found: {len(matching_columns)}")
        return True
    else:
        print(f"No columns contain '{string_to_check}'.")
        return False

# Example usage
# string_to_check = "temp_0.7_topP_0.8_topK_20"
# print(check_string_in_any_column(df_to_analyze, string_to_check))
```
   
