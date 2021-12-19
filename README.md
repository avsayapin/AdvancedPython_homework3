Repository for the third homework of the Advanced Python course

[![Build Status](https://app.travis-ci.com/avsayapin/AdvancedPython_homework3.svg?token=qxcp5FiEzzfpzyWqKeqT&branch=master)](https://app.travis-ci.com/avsayapin/AdvancedPython_homework3)

Information about API from previous homework can be found in "API information" section. 
Changes made to API reflected in the beginning of it.
# HW3 Tasks
____
<details><summary>1. Logging and monitoring</summary>
<p>

Logging was added via 'logging' library from base python. 

Logs are saved to app.log and worker.log files on the volume named "advancedpython_homework3_logs".

You can set logs level by changing logger.setlevel() in app/api.py and worker/tasks.py.

Prometheus from dockerhub image 'prom/prometheus' has been added for monitoring.

It can be accessed from standard address 'localhost:9090', metrics can be seen at 'localhost:5000/metrics'.

Added counters by model for training and prediction requests, counter by class for create_model request 
and counter by status code for results request.

Histogram type metrics added for classes and metrics requests.
</p>
</details>
<details><summary>2. MLFlow</summary>
<p>
Added MLFlow for model and metrics tracking, 
quick setup was made with sqlite(as backend) and volume "models" as artifact store.

MongoDB was deleted and celery tasks were changed to use MLFlow.

Also added "/test" method for API that returns value for provided metric argument. 
Supported metrics can be found in class Models.metrics from models.py.

MLFlow is running on port 5050, you can access it with "http://localhost:5050/"
</p>
</details>
<details><summary>3. CI</summary>
<p>

- Travis CI was used
- Docker images are made during build using docker-compose and then pushed to DockerHub
- You should specify DOCKER_USERNAME and DOCKER_PASSWORD for successful push
- Badges: build badge from Travis
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

# API information
___
Changes from the previous homework:


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
