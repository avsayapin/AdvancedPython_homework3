Repository for the first homework of the Advanced Python course
API can do the following requests:

**/classes**

returns available classes

**/models**

returns available models stored in 'models' directory

**/create_model**

args={name:name,class_name:class_name,params:params}

creates new model with specified arguments, "class name" arg is mandatory, params should be a string dictionary with hyperparameters of the chosen sklearn model class

Example:
```
params='{"learning_rate": 0.01,...}'
```

**/delete**

args={name:name}

deletes model with specified name

**/train**

args={"name":name},json

training data should be sent in json file and contain 'target' column, example using requests: 
```
requests.get(url,json=data)
```
Function was tested with pandas dataframe that has been converted to dict using:
```
dd.to_dict(orient="records")
```

**/predict**

args={name:name},json

data should be sent in the same format as in '/train'
