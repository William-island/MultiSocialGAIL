import os
import numpy as np
import torch
from .network import GraphData


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

class SerializedBuffer_SA:

    def __init__(self, path, device):
        print(f'Loading expert demo from {path}...')

        tmp = torch.load(path)
        self.buffer_size = self._n = len(tmp['state'])
        self.device = device

        print(f'Expert demo size: {self.buffer_size}')

        self.states = tmp['state']    #.clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        states = []
        for i in idxes:
            s = self.states[i]
            states.append(s)

        return (
            states,
            self.actions[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)




class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0         # number of all samples
        self._c = 0         # number of completed samples
        self.device = device

        self.states = {}
        self.actions = {}
        self.rewards = {}
        self.dones = {}
        self.log_pis = {}
        self.next_states = {}

        # self.states = [None for i in range(self.total_size)]
        # self.actions = torch.empty(
        #     (self.total_size, *action_shape), dtype=torch.float, device=device)
        # self.rewards = torch.empty(
        #     (self.total_size, 1), dtype=torch.float, device=device)
        # self.dones = torch.empty(
        #     (self.total_size, 1), dtype=torch.float, device=device)
        # self.log_pis = torch.empty(
        #     (self.total_size, 1), dtype=torch.float, device=device)
        # self.next_states = [None for i in range(self.total_size)]
        # self.next_states = torch.empty(
        #     (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, states, actions, rewards, dones, log_pis, next_states):
        for id in states.keys():
            if id not in self.states.keys():                           # create new key
                self.states[id] = [states[id].to(self.device)]
                self.actions[id] = [torch.from_numpy(actions[id])]
                self.rewards[id] = [float(rewards[id])]
                self.dones[id] = [float(dones[id])]
                self.log_pis[id] = [next_states[id].to(self.device)]
            else:                                                      # fill in existing key
                self.states[id].append(states[id].to(self.device))
                self.actions[id].append(torch.from_numpy(actions[id]))
                self.rewards[id].append(float(rewards[id]))
                self.dones[id].append(float(dones[id]))
                self.log_pis[id].append(next_states[id].to(self.device))

        self._n += len(states.keys())

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        states, next_states = [], []
        for i in idxes:
            s = self.states[i]
            ns = self.next_states[i]
            states.append(s)
            next_states.append(ns)

        return (
            states,
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            next_states
        )
    
    def clear(self):
        self.states = {}
        self.actions = {}
        self.rewards = {}
        self.dones = {}
        self.log_pis = {}
        self.next_states = {}

class SeparatedRolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, batch_size):
        self._n = 0
        self._p = 0     
        self.batch_size = batch_size
        self.total_size = buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        start = self.total_size-self.batch_size
        idxes = slice(start, self.total_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n-self.batch_size, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
