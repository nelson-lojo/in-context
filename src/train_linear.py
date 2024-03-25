from function_classes.linear import LinearRegression
from models.transformer import TransformerModel
from train.context_trainer import ContextTrainer, TrainerSteps
from parse import parse_training

import torch 
import wandb

wandb.init()

stages, model, yaml_str = parse_training("train_linear.yml")
model = model.to(torch.device("cuda:0"))
trainer = TrainerSteps(stages)
trainer.train()