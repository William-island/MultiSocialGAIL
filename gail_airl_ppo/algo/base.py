from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, states):
        # sample actions with noise
        actions = {}
        log_pis = {}
        for id, state in states.items():
            state = state.to(self.device)
            with torch.no_grad():
                action, log_pi = self.actor.sample(state)
            actions[id] = action.cpu().numpy()[0]
            log_pis[id] = log_pi.item()        
        return actions, log_pis

    def exploit(self, states):
        # sample action without noise
        actions = {}
        for id, state in states.items():
            state = state.to(self.device)
            with torch.no_grad():
                action = self.actor(state)       
            actions[id] = action.cpu().numpy()[0]   
        return actions

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
