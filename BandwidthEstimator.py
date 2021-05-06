#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import onnx, torch, onnxruntime
import numpy as np
from utils.packet_info import PacketInfo
from utils.packet_record import PacketRecord
from deep_rl.actor_critic import ActorCritic
import torch


UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)


def liner_to_log(value):
    # from 10kbps~8Mbps to 0~1
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    log_value = np.log(value)
    return (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M


class Estimator(object):
    def __init__(self, model_path, step_time=60):
        state_dim = 4
        action_dim = 1
        exploration_param = 0.05    # the std var of action distribution
        # load model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, action_dim, exploration_param, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.bandwidth_prediction = 1e6
        self.step_time = step_time

    def report_states(self, stats: dict):
        '''
        stats is a dict with the following items
        {
            "send_time_ms": uint,
            "arrival_time_ms": uint,
            "payload_type": int,
            "sequence_number": uint,
            "ssrc": int,
            "padding_length": uint,
            "header_length": uint,
            "payload_size": uint
        }
        '''
        packet_info = PacketInfo()
        packet_info.payload_type = stats["payload_type"]
        packet_info.ssrc = stats["ssrc"]
        packet_info.sequence_number = stats["sequence_number"]
        packet_info.send_timestamp = stats["send_time_ms"]
        packet_info.receive_timestamp = stats["arrival_time_ms"]
        packet_info.padding_length = stats["padding_length"]
        packet_info.header_length = stats["header_length"]
        packet_info.payload_size = stats["payload_size"]
        packet_info.bandwidth_prediction = self.bandwidth_prediction

        self.packet_record.on_receive(packet_info)

        # calculate state
        states = []
        receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        states.append(liner_to_log(receiving_rate))
        delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        states.append(min(delay/1000, 1))
        loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        states.append(loss_ratio)
        latest_prediction = self.packet_record.calculate_latest_prediction()
        states.append(liner_to_log(latest_prediction))

        torch_tensor_states = torch.Tensor(states)
        # onnx_input = {self.model.get_inputs()[0].name : torch_tensor_states}
        print(torch_tensor_states)
        action, action_logprobs, value = self.model.forward(torch_tensor_states)
        print(action, action_logprobs, value)
        # self.bandwidth_prediction = liner_to_log(action)
        self.bandwidth_prediction = log_to_linear(action)

    def get_estimated_bandwidth(self)->int:

        return self.bandwidth_prediction



def test_torch_model(model_path):
    lr = 3e-5                 # Adam parameters
    betas = (0.9, 0.999)
    state_dim = 4
    action_dim = 1
    exploration_param = 0.05    # the std var of action distribution
    K_epochs = 37               # update policy for K_epochs
    ppo_clip = 0.2              # clip parameter of PPO
    gamma = 0.99                # discount factor

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(state_dim, action_dim, exploration_param, device).to(device)
    policy.load_state_dict(torch.load(model_path))

    state = np.array([[0.5, 0.2, 0.3, 0.4]])
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    action, action_logprobs, value = policy.forward(state)
    print(action)


if __name__ == "__main__":
    model_path = "./pretrained_model.pth"
    BWE = Estimator(model_path)
    states = {
            "send_time_ms": 100,
            "arrival_time_ms": 400,
            "payload_type": 125,
            "sequence_number": 10,
            "ssrc": 123,
            "padding_length": 0,
            "header_length": 120,
            "payload_size": 1350
        }
    print(BWE.get_estimated_bandwidth())
    print(BWE.report_states(states))
    print(BWE.get_estimated_bandwidth())
    
    # test_torch_model(model_path)