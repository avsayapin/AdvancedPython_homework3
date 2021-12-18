Repository for the third homework of the Advanced Python course
# HW3 Tasks
____
<details><summary>1. Logging and monitoring</summary>
<p>

Logging was added via 'logging' library from base python. 
Logs are saved to app.log file on the volume named 'advancedpython_homework3_logs'
Prometheus from dockerhub image 'prom/prometheus' has been added for monitoring
It can be accessed from standard address 'localhost/9090'
</p>
</details>
<details><summary>2. MLFlow</summary>
<p>

</p>
</details>
<details><summary>3. CI</summary>
<p>

</p>
</details>
<details><summary>4. Telegram app</summary>
<p>

</p>
</details>
<details><summary>5. Extra celery email task</summary>
<p>

</p>
</details>

___
# API information
___
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
