## Execution

- Ensure environment is setup correctly.  
- Run preprocess.py to preprocess the data.
- Run nn.py to train and test the neural network  
- Alternatively, load weights of trained models are from the checkpoints directory. The final checkpoints are of the form:  
`depth x width_activation-fn_loss-fn_batch_[epochs]_loss.hdf5`

## Setup

### Create virtual env

`virtualenv --system-site-packages -p python3 ./nn_venv`  

Activate it:  
`source ./nn_venv/bin/activate `

### Setup dependencies 

Ensure keras is installed:  
`pip3 install keras`  

Ensure tensorflow is installed:  
`pip3 install --upgrade tensorflow-gpu`  

ENsure sklearn is installed:  
`pip3 install sklearn`  

Install xgboost:  
`pip3 install xgboost`