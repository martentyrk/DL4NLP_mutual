# A study of methodologies to improve BERT performance on dialogue modelling

## Abstract 
With the rise of transformer-based models in various conversational tasks, their limitations in reasoning abilities have become more apparent. This study aims to investigate how different design choices affect a model's reasoning capabilities. We conducted experiments using both BERT, a general text pre-trained model, and TOD-BERT, a model specifically designed for dialogues. One noteworthy enhancement we introduce is the incorporation of two loss functions into the traditional BERT model: a contrastive learning-based regularizer (CL) and a correlation matrix-based regularizer (COR). We explore the individual and combined effects of these enhancements on model performance. Our evaluation was conducted on the MuTual dataset, which is well-known for assessing logical reasoning skills. We aim to shed some light on the nuances of transformer models' performance in reasoning tasks and show the importance of design intricacies in enhancing their capabilities.


## Setup
Please use the environment.yml file to set up your environment.

When using a conda environment, this can be done by running
```
conda env create -f environment.yml
```

In bert.py we have specified two options for logging the experiments, namely Comet and wandb. In order to run the experiments with Comet, please create a .env file in the root of the directory, and include the following in the file:
```
COMET_API_KEY="your_api_key"
WORKSPACE="your_workspace_name"
```
To log all data with wandb, we suggest to follow the instructions given [here](https://docs.wandb.ai/quickstart).

## Dataset
In this project we use the MuTual dataset, which can be found [here](https://github.com/Nealcly/MuTual/tree/master).

## Running the experiments
In order to run the experiments, you can use the following snippet:

```
python main.py \
    --max_epochs 50 \
    --freeze_lm \
    --data_dir 'insert path to mutual data folder here, for example data/mutual' \
    --device 'cuda' \
    --model_name 'bert'
```
or just use the run_experiments.sh file we've provided under the scripts folder.
### Model choice
In order to run the different experiments, you can use the --model_name to choose between the models (tod-bert or bert).
### Regularizers
For experiments regarding regularizers, we have specified the parameters
--use_contrastive, --use_correlation to activate one or the other or both.
 ### Data preprocessing
To run the experiment we did with data preprocessing, please use the 
--A_plus argument to do so.


Note: All of our changeble parameters have also been specified in main.py with descriptions to their purpose.