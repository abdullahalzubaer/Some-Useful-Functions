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
