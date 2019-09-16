#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation. All rights reserved.
# 
# Licensed under the MIT License.

# ## Introduction to Azure Machine Learning: Deploy web service
# 
# In previous example, you ran an experiment to estimate value of pi. In this example, we'll use your estimated value to create a web service that computes the area of a circle in a real time. You'll learn about following concepts:
# 
# **Model** is simply a file - or a folder of files - that model management service tracks and versions. Typically a model file would contain the coefficients of your trained model, saved to a file.  
# 
# **Image** is a combination of model, a Python scoring script that tells how to load and invoke the model, and Python libraries used to execute that code. It is a self-contained unit that can be deployed as a service.
# 
# **Service** is the image running on a compute. The service can be called from your front-end application to get predictions, either using the Azure ML SDK or raw HTTP.

# **Important**: This notebook uses Azure Container Instances (ACI) as the compute for the service. If you haven't registered ACI provider with your Azure subscription, run the following 2 cells first. Note that you must be the administrator of your Azure subscription to register a provider.

# In[ ]:


get_ipython().system(u'az provider show -n Microsoft.ContainerInstance -o table')


# In[ ]:


get_ipython().system(u'az provider register -n Microsoft.ContainerInstance')


# Let's load the workspace, and retrieve the latest run from your experiment using *Experiment.get_runs* method.

# In[ ]:


from azureml.core import Workspace, Experiment, Run
import math, random, pickle, json


# In[ ]:


ws = Workspace.from_config()


# In[ ]:


experiment_name = "my-first-experiment"
run = list(Experiment(workspace = ws, name = experiment_name).get_runs())[0]


# In the previous example you saved a file containing the pi value into run history. Registering the file makes it into a model that is tracked by Azure ML model management.

# In[ ]:


model = run.register_model(model_name = "pi_estimate", model_path = "outputs/pi_estimate.txt")


# Let's create a scoring script that computes an area of a circle, given the estimate within the pi_estimate model. The scoring script consists of two parts: 
# 
#  * The *init* method that loads the model. You can retrieve registered model using *Model.get_model_path* method. 
#  * The *run* method that gets invoked when you call the web service. It computes the area of a circle using the well-known $area = \pi*radius^2$ formula. The inputs and outputs are passed as json-formatted strings.

# In[ ]:


get_ipython().run_cell_magic(u'writefile', u'score.py', u'import pickle, json\nfrom azureml.core.model import Model\n\ndef init():\n    global pi_estimate\n    model_path = Model.get_model_path(model_name = "pi_estimate")\n    with open(model_path, "rb") as f:\n        pi_estimate = float(pickle.load(f))\n\ndef run(raw_data):\n    try:\n        radius = json.loads(raw_data)["radius"]\n        result = pi_estimate * radius**2\n        return json.dumps({"area": result})\n    except Exception as e:\n        result = str(e)\n        return json.dumps({"error": result})')


# You also need to specify the library dependencies of your scoring script as conda yml file. This example doesn't use any special libraries, so let's simply use Azure ML's default dependencies.

# In[ ]:


from azureml.core.conda_dependencies import CondaDependencies 

cd = CondaDependencies()
cd.save_to_file(".", "myenv.yml")


# Then, let's deploy the web service on Azure Container Instance: a serverless compute for running Docker images. Azure ML service takes care of packaging your model, scoring script and dependencies into Docker image and deploying it.

# In[ ]:


from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.image import ContainerImage

# Define the configuration of compute: ACI with 1 cpu core and 1 gb of memory.
aci_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

# Specify the configuration of image: scoring script, Python runtime (PySpark is the other option), and conda file of library dependencies.
image_config = ContainerImage.image_configuration(execution_script = "score.py", 
                                    runtime = "python", 
                                    conda_file = "myenv.yml")

# Deploy the web service as an image containing the registered model.
service = Webservice.deploy_from_model(name = "area-calculator",
                                       deployment_config = aci_config,
                                       models = [model],
                                       image_config = image_config,
                                       workspace = ws)

# The service deployment can take several minutes: wait for completion.
service.wait_for_deployment(show_output = True)


# You can try out the web service by passing in data as json-formatted request. Run the cell below and move the slider around to see real-time responses.

# In[ ]:


from ipywidgets import interact

def get_area(radius):
    request = json.dumps({"radius": radius})
    response = service.run(input_data = request)
    return json.loads(response)["area"]

interact(get_area,radius=(0,10))


# Finally, delete the web service once you're done, so it's not consuming resources.

# In[ ]:


service.delete()


# As your next step, take a look at the more detailed tutorial for building an image classification model using Azure Machine Learning service.
# 
# [tutorials/img-classification-part1-training](./tutorials/img-classification-part1-training.ipynb)

# In[ ]:




