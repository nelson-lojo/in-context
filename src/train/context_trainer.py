from core import FunctionClass
from torch.optim import Optimizer
from core import ContextModel
import torch
from torch import nn

import wandb
from typing import Optional, List, Any

class ContextTrainer:
    def __init__(
        self, 
        function_class: FunctionClass,
        model: ContextModel,
        optim: Optimizer, 
        loss_fn: nn.Module,
        steps: int,
        baseline_models: List,
        log_freq: int = -1,
        **kwargs
    ):
        self.func_class = function_class
        self.model = model
        self.optimizer = optim 
        self.loss_func = loss_fn
        self.num_steps = steps
        self.baseline_models = baseline_models
        self.log_freq = log_freq 
        # wandb.log(self.metadata)
        #self.metadata = ... # config stuff here ********* TODO: excuse me what



    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        baseline_loss = {}

        for i, (x_batch, y_batch) in zip(range(self.num_steps), self.func_class):
            
            output = self.model(x_batch, y_batch)
            loss = self.loss_func(output, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"loss {loss}")

            if i % self.log_freq == 0:

                for baseline in self.baseline_models:
                    baseline_output = baseline(x_batch, y_batch)
                    with torch.no_grad():
                        baseline_loss[baseline.name] = self.loss_func(baseline_output, y_batch)

                log_dict = {
                        "overall_loss": loss,
                    } 
                log_dict |= {f"baseline_loss_{baseline.name}": baseline_loss[baseline.name] for baseline in self.baseline_models}

                # wandb.log(
                #     data=log_dict,
                #     step=i,
                # )
                wandb.log(
                    data=log_dict
                )

            # TODO: stretch goal: log functions / precompute dataset?
            # TODO: log the xs, ys?

        return self.model

    # @property
    # def metadata(self) -> dict:
    #     return self.metadata
class TrainerSteps(ContextTrainer):

    def __init__(self, 
        trainers: list[ContextTrainer]
    ):
        self.trainers = trainers

    def train(self, pbar: Optional[Any] = None) -> ContextModel:

        for trainer in self.trainers:
            self.model = trainer.train(pbar)

        return self.model
