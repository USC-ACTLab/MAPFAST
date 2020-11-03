# MAPFAST

This repository contains the official implementation for our paper "MAPFAST: A Deep Algorithm Selector for Multi Agent Path Finding using Shortest Path Embeddings"

## After cloning the repo
```
cd MAPFAST
pip3 install -r requirements.txt
```
This will install all the packages necessary for training/testing the model.

## Dataset

Following items are needed for training the model:
```
1. yaml_details -> Json file which contains file name as key and another json object(which has 'SOLVER' storing the name of the fastest solver and solving time(in seconds) for all the solvers) as value.
2. agent_details -> Json file which contains file name as key and its value is a json object containing the start and goal locations of the agents.
3. map_details -> Json file which contains file name as key and another json object(which has number of agents, dimensions of the input map) as value.
4. .png or .npz files -> Images or compressed numpy representations of the map which will be given as input to CNN. Note that the names of these images/numpy files should match the name in yaml_details/agent_details/map_details.
```

## MAPFAST

MAPFAST.py contains the necessary class for training and testing the model as described in our paper.

### Training model

Following shows a way to train a model with the default setting given the necessary json files.

```
mapfast = MAPFAST(device, yaml_details, agent_details, map_details, input_location, mapping)

train_list, valid_list, test_list = mapfast.get_train_valid_test_list()

model = mapfast.train_model(train_list, valid_list, model_loc, model_name)
```

This will train the default MAPFAST model which has 14 output neurons(4 for best solver classification, 4 for finish prediction and 6 for pairwise comparison).
Here, device is the device type in which we want to train.

### Testing model

Following shows a way to test a model with the default setting given the necessary json files and model location details.

```
mapfast = MAPFAST(device, yaml_details, agent_details, map_details, input_location, mapping)

train_list, valid_list, test_list = mapfast.get_train_valid_test_list()

prediction_data = mapfast.test_model(test_list, model_loc, model_name)
```

This will test the default MAPFAST model at given location and given name and return a json object with the test details.

### Note
When creating the MAPFAST class, we have to give a json object called mapping which maps the solver name to a unique number.
Example mapping is shown below.
```
{"BCP":0, "CBS":1, "CBSH":2, "SAT":3}
```

## New Dataset

To train the model with new dataset and potentially with different solvers, the necessary yaml_details, agent_details, map_details json files and the input(image/numpy) has to be set properly. The mapping json has to be changed to reflect the solvers in the portfolio. No other changes inside the code is necessary to train with a new dataset.

## Misc Info

1. main.py provides a sample implementation where all the parameters are fetched from a config.json file. The location of config.json can also be passed as a command line argument to the python script. The default location is json_files folder.

2. analysis.py prints a detailed report which includes the accuracy, coverage and custom score details as described in our paper.

3. Use dir(MAPFAST) or help(MAPFAST) to get information and usage details for each function in MAPFAST and InceptionClassificationNet class.