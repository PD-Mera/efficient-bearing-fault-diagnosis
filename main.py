import os
from pathlib import Path

import torch
import nni.retiarii.strategy as strategy
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
from nni.retiarii.evaluator import FunctionalEvaluator
# import pytorch_lightning as pl
import nni.retiarii.evaluator.pytorch.lightning as pl
from torch.utils.data import ConcatDataset

from config import *
from data import *
from model import *
from exec_function import evaluate_model

model_space = BaseModelSpace()
print(model_space)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

search_strategy = strategy.Random(dedup = False)  # dedup=False if deduplication is not wanted

# Multi-trial
evaluator = FunctionalEvaluator(evaluate_model)

exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'Bearing_fault_diagnosis'
exp_config.max_trial_number = 500   # spawn 4 trials at most
exp_config.trial_concurrency = 4  # will run two trials concurrently
exp_config.trial_gpu_number = 1
exp_config.training_service.use_active_gpu = True
export_formatter = 'code'


exp.run(exp_config, 8081)

for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)

