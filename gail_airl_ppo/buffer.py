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

    def __init__(self, buffer_size, device):
        self._n = 0         # number of all samples
        self._c = 0         # number of completed samples
        self.buffer_size = buffer_size
        self.device = device

        self.all_states = {}
        self.all_actions = {}
        self.all_rewards = {}
        self.all_dones = {}
        self.all_log_pis = {}
        self.all_next_states = {}

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_pis = []
        self.next_states = []

    def append(self, states, actions, rewards, dones, log_pis, next_states):
        for id in states.keys():
            if id not in self.all_states.keys():                           # create new key
                self.all_states[id] = [states[id].to(self.device)]
                self.all_actions[id] = [actions[id]]
                self.all_rewards[id] = [[float(rewards[id])]]
                self.all_dones[id] = [[float(dones[id])]]
                self.all_log_pis[id] = [[float(log_pis[id])]]
                self.all_next_states[id] = [next_states[id].to(self.device)]
            else:                                                      # fill in existing key
                self.all_states[id].append(states[id].to(self.device))
                self.all_actions[id].append(actions[id])
                self.all_rewards[id].append([float(rewards[id])])
                self.all_dones[id].append([float(dones[id])])
                self.all_log_pis[id].append([float(log_pis[id])])
                self.all_next_states[id].append(next_states[id].to(self.device))
            # move trajectories that completed to the buffer
            if dones[id] and len(self.all_actions[id])>1: #& len(self.all_actions[id])>1
                self.states.extend(self.all_states[id])
                del self.all_states[id]
                self.actions.extend(self.all_actions[id])
                del self.all_actions[id]
                self.rewards.extend(self.all_rewards[id])
                del self.all_rewards[id]
                self.dones.extend(self.all_dones[id])
                del self.all_dones[id]
                self.log_pis.extend(self.all_log_pis[id])
                del self.all_log_pis[id]
                self.next_states.extend(self.all_next_states[id])
                del self.all_next_states[id]

        self._n += len(states.keys())
        self._c = len(self.states)  # current buffer length

    def is_full(self):
        return self._c >= self.buffer_size

    def get(self):
        assert self._c >= self.buffer_size
        idxes = slice(0, self.buffer_size)
        return (
            self.states[idxes],
            torch.tensor(np.array(self.actions),dtype=torch.float, device=self.device)[idxes],
            torch.tensor(np.array(self.rewards),dtype=torch.float, device=self.device)[idxes],
            torch.tensor(np.array(self.dones),dtype=torch.float, device=self.device)[idxes],
            torch.tensor(np.array(self.log_pis),dtype=torch.float, device=self.device)[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._c >= self.buffer_size
        idxes = np.random.randint(low=0, high=self.buffer_size, size=batch_size)

        states, next_states = [], []
        for i in idxes:
            s = self.states[i]
            ns = self.next_states[i]
            states.append(s)
            next_states.append(ns)

        return (
            states,
            torch.tensor(np.array(self.actions),dtype=torch.float, device=self.device)[idxes],
            torch.tensor(np.array(self.rewards),dtype=torch.float, device=self.device)[idxes],
            torch.tensor(np.array(self.dones),dtype=torch.float, device=self.device)[idxes],
            torch.tensor(np.array(self.log_pis),dtype=torch.float, device=self.device)[idxes],
            next_states
        )
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_pis = []
        self.next_states = []



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
