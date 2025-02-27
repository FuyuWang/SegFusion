import math

from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Transformer


class Actor(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, accelerator, buf_spmap_cstr, buffer_size_list, steps_per_level,
                 problem_instance, prime2idx, tile_budgets, pre_tile_budgets):
        super(Actor, self).__init__()

        self.accelerator = accelerator
        self.prime2idx = prime2idx
        self.idx2prime = {value: key for key, value in prime2idx.items()}
        self.num_primes = len(self.prime2idx.keys())

        self.tile_budgets = tile_budgets
        self.pre_tile_budgets = pre_tile_budgets

        self.transformer = Transformer(d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
                                       n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=0,
                                       n_position=100, trg_emb_prj_weight_sharing=True,
                                       scale_emb_or_prj='prj',
                                       order_size=steps_per_level, tile_size=30,
                                       num_primes=len(prime2idx.keys()))
        self.buffer_size_list = buffer_size_list
        self.buf_spmap_cstr = buf_spmap_cstr
        self.steps_per_level = steps_per_level
        self.problem_instance = problem_instance
        self.finished_levels = []

    def reset(self):
        self.finished_levels = []

    def get_remain_buffer_size(self, cur_buffer_level, program_seq_disorder, loop_ind):
        if 'simba' in self.accelerator:
            return self.get_remain_buffer_size_for_simba(cur_buffer_level, program_seq_disorder, loop_ind)
        elif 'Eyeriss' in self.accelerator:
            return self.get_remain_buffer_size_for_eyeriss(cur_buffer_level, program_seq_disorder, loop_ind)
        elif 'nvdla' in self.accelerator:
            return self.get_remain_buffer_size_for_nvdla(cur_buffer_level, program_seq_disorder, loop_ind)

    def get_remain_buffer_size_for_simba(self, cur_buffer_level, program_seq_disorder, loop_ind):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = program_seq_disorder.size(0)
        tiles = program_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_program_seq_disorder[:, :, v + 1])

        R, S, P, Q, C, K, H, N = torch.unbind(tiles, dim=1)
        wstride = self.problem_instance['Wstride']
        hstride = self.problem_instance['Hstride']
        wdilation = self.problem_instance['Wdilation']
        hdilation = self.problem_instance['Hdilation']
        if cur_buffer_level == 1 or cur_buffer_level == 3:  # pe reg/weight
            N = program_seq_disorder.new_zeros(batch_size)
            P = program_seq_disorder.new_zeros(batch_size)
            Q = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # pe acc
            C = program_seq_disorder.new_zeros(batch_size)
            if self.problem_instance['type'] == 'C2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 4:  # pe input
            K = program_seq_disorder.new_zeros(batch_size)
            if self.problem_instance['type'] == 'T2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)

        if self.problem_instance['type'] == 'C2D':
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            weight_tile = K * R * S * C * H
            output_tile = P * Q * K * N * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
            K_coef = (R * S * C + P * Q * N) * K * H
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
            H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C + P * Q * K * N) * H
        elif self.problem_instance['type'] == 'T2D':
            input_tile = P * Q * C * N * H
            weight_tile = K * R * S * C * H
            output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                    (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                    (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
            K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + C * R * S) * K * H
            C_coef = (P * Q * N + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + K * R * C) * S * H
            H_coef = (P * Q * C * N + K * R * S * C + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        if cur_buffer_level == 5:  # global buffer
            if self.problem_instance['type'] == 'C2D':
                input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = P * Q * K * N * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
                K_coef = P * Q * N * K * H
                C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * S * H
                H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K * N) * H
            elif self.problem_instance['type'] == 'T2D':
                input_tile = P * Q * C * N * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
                K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * K * H
                C_coef = (P * Q * N) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K) * S * H
                H_coef = (P * Q * C * N + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        else:
            if cur_buffer_level == 1 or cur_buffer_level == 3:  # pe/reg weight
                N = program_seq_disorder.new_zeros(batch_size)
                P = program_seq_disorder.new_zeros(batch_size)
                Q = program_seq_disorder.new_zeros(batch_size)

                N_sub = program_seq_disorder.new_zeros(batch_size).float()
                P_sub = program_seq_disorder.new_zeros(batch_size).float()
                Q_sub = program_seq_disorder.new_zeros(batch_size).float()
                N_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:  # pe acc
                C = program_seq_disorder.new_zeros(batch_size)
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.problem_instance['type'] == 'C2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

            elif cur_buffer_level == 4:  # pe input
                K = program_seq_disorder.new_zeros(batch_size)
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.problem_instance['type'] == 'T2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([R_coef, S_coef, P_coef, Q_coef, C_coef, K_coef, H_coef, N_coef], dim=1)[:, loop_ind]
        sub_arr = torch.stack([R_sub, S_sub, P_sub, Q_sub, C_sub, K_sub, H_sub, N_sub], dim=1)[:, loop_ind]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size

    def get_remain_buffer_size_for_eyeriss(self, cur_buffer_level, program_seq_disorder, loop_ind):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = program_seq_disorder.size(0)
        tiles = program_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_program_seq_disorder[:, :, v + 1])

        R, S, P, Q, C, K, H, N = torch.unbind(tiles, dim=1)
        wstride = self.problem_instance['Wstride']
        hstride = self.problem_instance['Hstride']
        wdilation = self.problem_instance['Wdilation']
        hdilation = self.problem_instance['Hdilation']
        if cur_buffer_level == 1:  # pe acc
            C = program_seq_disorder.new_zeros(batch_size)
            if self.problem_instance['type'] == 'C2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # pe weight
            N = program_seq_disorder.new_zeros(batch_size)
            P = program_seq_disorder.new_zeros(batch_size)
            Q = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 3:  # pe input
            K = program_seq_disorder.new_zeros(batch_size)
            if self.problem_instance['type'] == 'T2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 4:  # dummybuffer
            H = program_seq_disorder.new_zeros(batch_size)
            N = program_seq_disorder.new_zeros(batch_size)
            K = program_seq_disorder.new_zeros(batch_size)
            C = program_seq_disorder.new_zeros(batch_size)
            P = program_seq_disorder.new_zeros(batch_size)
            Q = program_seq_disorder.new_zeros(batch_size)
            R = program_seq_disorder.new_zeros(batch_size)
            S = program_seq_disorder.new_zeros(batch_size)

        if self.problem_instance['type'] == 'C2D':
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            weight_tile = K * R * S * C * H
            output_tile = P * Q * K * N * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
            K_coef = (R * S * C + P * Q * N) * K * H
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
            H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C + P * Q * K * N) * H
        elif self.problem_instance['type'] == 'T2D':
            input_tile = P * Q * C * N * H
            weight_tile = K * R * S * C * H
            output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                    (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                    (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
            K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + C * R * S) * K * H
            C_coef = (P * Q * N + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + K * R * C) * S * H
            H_coef = (P * Q * C * N + K * R * S * C + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        if cur_buffer_level == 5:  # global buffer
            if self.problem_instance['type'] == 'C2D':
                input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = P * Q * K * N * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
                K_coef = P * Q * N * K * H
                C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * S * H
                H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K * N) * H
            elif self.problem_instance['type'] == 'T2D':
                input_tile = P * Q * C * N * H
                weight_tile = program_seq_disorder.new_zeros(batch_size)
                output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
                K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1))) * K * H
                C_coef = (P * Q * N) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K) * S * H
                H_coef = (P * Q * C * N + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        else:
            if cur_buffer_level == 1:  # pe acc
                C = program_seq_disorder.new_zeros(batch_size)
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.problem_instance['type'] == 'C2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:  # pe weight
                N = program_seq_disorder.new_zeros(batch_size)
                P = program_seq_disorder.new_zeros(batch_size)
                Q = program_seq_disorder.new_zeros(batch_size)
                N_sub = program_seq_disorder.new_zeros(batch_size).float()
                P_sub = program_seq_disorder.new_zeros(batch_size).float()
                Q_sub = program_seq_disorder.new_zeros(batch_size).float()
                N_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 3:  # pe input
                K = program_seq_disorder.new_zeros(batch_size)
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.problem_instance['type'] == 'T2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 4:  # dummybuffer
                H_sub = program_seq_disorder.new_zeros(batch_size).float()
                N_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                P_sub = program_seq_disorder.new_zeros(batch_size).float()
                Q_sub = program_seq_disorder.new_zeros(batch_size).float()
                R_sub = program_seq_disorder.new_zeros(batch_size).float()
                S_sub = program_seq_disorder.new_zeros(batch_size).float()
                H_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                N_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                P_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                Q_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([R_coef, S_coef, P_coef, Q_coef, C_coef, K_coef, H_coef, N_coef], dim=1)[:, loop_ind]
        sub_arr = torch.stack([R_sub, S_sub, P_sub, Q_sub, C_sub, K_sub, H_sub, N_sub], dim=1)[:, loop_ind]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size

    def get_remain_buffer_size_for_nvdla(self, cur_buffer_level, program_seq_disorder, loop_ind):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = program_seq_disorder.size(0)
        tiles = program_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_program_seq_disorder[:, :, v + 1])

        R, S, P, Q, C, K, H, N = torch.unbind(tiles, dim=1)
        wstride = self.problem_instance['Wstride']
        hstride = self.problem_instance['Hstride']
        wdilation = self.problem_instance['Wdilation']
        hdilation = self.problem_instance['Hdilation']
        if cur_buffer_level == 1:  # LRF
            K = program_seq_disorder.new_zeros(batch_size)
            if self.problem_instance['type'] == 'T2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:  # RF
            C = program_seq_disorder.new_zeros(batch_size)
            if self.problem_instance['type'] == 'C2D':
                R = program_seq_disorder.new_zeros(batch_size)
                S = program_seq_disorder.new_zeros(batch_size)

        if self.problem_instance['type'] == 'C2D':
            input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            weight_tile = K * R * S * C * H
            output_tile = P * Q * K * N * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
            S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + P * Q * K) * N * H
            K_coef = (R * S * C + P * Q * N) * K * H
            C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + Q * K * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + P * K * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
            H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C + P * Q * K * N) * H
        elif self.problem_instance['type'] == 'T2D':
            input_tile = P * Q * C * N * H
            weight_tile = K * R * S * C * H
            output_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            N_sub = weight_tile
            K_sub = input_tile
            C_sub = output_tile

            P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                    (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            R_sub = input_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
            S_sub = input_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                    (P - 1) * wstride + 1 + wdilation * (R - 1)) * K * H
            H_sub = program_seq_disorder.new_zeros(batch_size).float()

            N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + P * Q * C) * N * H
            K_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) + C * R * S) * K * H
            C_coef = (P * Q * N + K * R * S) * C * H
            P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + Q * C * N) * P * H
            Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + P * C * N) * Q * H
            R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * K + K * S * C) * R * H
            S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * K + K * R * C) * S * H
            H_coef = (P * Q * C * N + K * R * S * C + N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                    (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K) * H
        if cur_buffer_level == 3:  # SMEM
            if self.problem_instance['type'] == 'C2D':
                input_tile = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                weight_tile = K * R * S * C * H
                output_tile = program_seq_disorder.new_zeros(batch_size)
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile + N * (1 - wstride + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                Q_sub = weight_tile + N * (1 - hstride + hdilation * (S - 1)) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                R_sub = output_tile + N * ((P - 1) * wstride + 1 - wdilation) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
                S_sub = output_tile + N * ((Q - 1) * hstride + 1 - hdilation) * (
                        (P - 1) * wstride + 1 + wdilation * (R - 1)) * C * H
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = (((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * N * H
                K_coef = (R * S * C) * K * H
                C_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) + K * R * S) * C * H
                P_coef = (N * wstride * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C) * P * H
                Q_coef = (N * hstride * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C) * Q * H
                R_coef = (N * wdilation * ((Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * S * C) * R * H
                S_coef = (N * hdilation * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * C + K * R * C) * S * H
                H_coef = (N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                        (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C + K * R * S * C) * H
            elif self.problem_instance['type'] == 'T2D':
                input_tile = P * Q * C * N * H
                weight_tile = K * R * S * C * H
                output_tile = program_seq_disorder.new_zeros(batch_size)
                N_sub = weight_tile
                K_sub = input_tile
                C_sub = output_tile

                P_sub = weight_tile
                Q_sub = weight_tile
                R_sub = input_tile
                S_sub = input_tile
                H_sub = program_seq_disorder.new_zeros(batch_size).float()

                N_coef = P * Q * C * N * H
                K_coef = C * R * S * K * H
                C_coef = (P * Q * N + K * R * S) * C * H
                P_coef = Q * C * N * P * H
                Q_coef = P * C * N * Q * H
                R_coef = K * S * C * R * H
                S_coef = K * R * C * S * H
                H_coef = (P * Q * C * N + K * R * S * C) * H
        else:
            if cur_buffer_level == 1:  # LRF
                K = program_seq_disorder.new_zeros(batch_size)
                K_sub = program_seq_disorder.new_zeros(batch_size).float()
                K_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.problem_instance['type'] == 'T2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
            elif cur_buffer_level == 2:  # pe acc
                C = program_seq_disorder.new_zeros(batch_size)
                C_sub = program_seq_disorder.new_zeros(batch_size).float()
                C_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                if self.problem_instance['type'] == 'C2D':
                    R = program_seq_disorder.new_zeros(batch_size)
                    S = program_seq_disorder.new_zeros(batch_size)
                    R_sub = program_seq_disorder.new_zeros(batch_size).float()
                    S_sub = program_seq_disorder.new_zeros(batch_size).float()
                    R_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)
                    S_coef = program_seq_disorder.new_ones(batch_size).float().fill_(1e-12)

        coef_arr = torch.stack([R_coef, S_coef, P_coef, Q_coef, C_coef, K_coef, H_coef, N_coef], dim=1)[:, loop_ind]
        sub_arr = torch.stack([R_sub, S_sub, P_sub, Q_sub, C_sub, K_sub, H_sub, N_sub], dim=1)[:, loop_ind]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size

    def get_remain_buf_spmap(self, cur_buffer_level, program_seq_disorder):
        buf_spmap_cstr = self.buf_spmap_cstr[f'l{cur_buffer_level}']
        start_ind = (cur_buffer_level - 1) * self.steps_per_level
        end_ind = cur_buffer_level * self.steps_per_level
        level_program_seq_disorder = copy.deepcopy(program_seq_disorder[:, start_ind:end_ind])
        num_samples = program_seq_disorder.size(0)
        used_buf_spmap = program_seq_disorder.new_ones(num_samples)
        # print(buf_spmap_cstr)
        for i in range(self.steps_per_level):
            sp_tile2 = level_program_seq_disorder[:, i, self.num_primes + 1]
            used_buf_spmap *= torch.clamp(torch.pow(2, sp_tile2), min=1)
        remain_buf_spmap = buf_spmap_cstr / used_buf_spmap.float()

        return remain_buf_spmap

    def get_max_temporal_size(self, cur_buffer_level, tile2_remain_dimension_budgets, remain_buf_spmap):
        '''
        param: tile2_remain_budget [batch, 7]
        '''
        # print(tile2_remain_budget.size(), tile2_remain_budget[0])

        max_temporal_tile2 = tile2_remain_dimension_budgets - torch.log2(torch.clamp(remain_buf_spmap, min=1))

        for level in range(1, len(self.buffer_size_list) + 1):
            buf_spmap_cstr = self.buf_spmap_cstr[f'l{level}']
            if level not in self.finished_levels and level != cur_buffer_level:
                max_temporal_tile2 -= math.log2(buf_spmap_cstr)
        # if cur_buffer_level == 5:
        #     print(tile2_remain_budget[0], remain_buf_spmap[0], max_temporal_tile2[0])
        return torch.clamp(max_temporal_tile2, min=0).long()

    def get_spatial_size(self, mode, cur_buffer_level, tile2_max, loop_ind,
                         tile2_remain_budget, remain_buf_spmap):
        sp_tile2_max = torch.log2(torch.clamp(remain_buf_spmap, min=1))
        sp_tile2_max = torch.clamp(sp_tile2_max.long(), min=0)
        sp_tile2_max = torch.minimum(sp_tile2_max, tile2_max)

        if mode % self.steps_per_level == self.steps_per_level - 1:
            sp_tile2_min = sp_tile2_max
            self.finished_levels.append(cur_buffer_level)
        else:
            tile2_remain_dimension_budgets = (tile2_remain_budget[:, loop_ind + 1:]).sum(dim=-1)
            sp_tile2_min = torch.clamp(
                torch.log2(torch.clamp(remain_buf_spmap, min=1)) - tile2_remain_dimension_budgets, min=0)
            sp_tile2_min = torch.minimum(sp_tile2_min.long(), sp_tile2_max)
        return sp_tile2_max, sp_tile2_min

    def forward(self, program_seq, order_mask, tile_remain_budgets, tile_masks, mode, cur_buffer_level,
                program_seq_disorder):

        loop_ind = mode % self.steps_per_level

        remain_buffer_size = self.get_remain_buffer_size(cur_buffer_level, program_seq_disorder, loop_ind)
        for later_level in range(cur_buffer_level + 1, len(self.buffer_size_list) + 1):
            remain_buffer_size = torch.minimum(remain_buffer_size,
                                               self.get_remain_buffer_size(later_level, program_seq_disorder, loop_ind))
        tile2_max = torch.log2(torch.clamp(remain_buffer_size, min=1))
        tile2_max = torch.clamp(tile2_max.long(), min=0)
        tile2_remain_budgets = tile_remain_budgets[:, :, 0]
        tile2_max = torch.minimum(tile2_max, tile2_remain_budgets[:, loop_ind])

        remain_buf_spmap = self.get_remain_buf_spmap(cur_buffer_level, program_seq_disorder)
        tile2_remain_dimension_budgets = tile2_remain_budgets.sum(dim=-1)
        max_temporal_tile2 = self.get_max_temporal_size(cur_buffer_level, tile2_remain_dimension_budgets, remain_buf_spmap)

        sp_tile2_max, sp_tile2_min = self.get_spatial_size(mode, cur_buffer_level, tile2_max, loop_ind,
                                                           tile2_remain_budgets, remain_buf_spmap)

        order_logit, tile_logits, sp_tile2_logit = self.transformer(program_seq)
        tile2_logit = tile_logits[:, 0]
        num_samples = program_seq.size(0)

        # order_action = program_seq.new_ones(num_samples).fill_(loop_ind)
        # order_log_prob = tile2_logit.new_zeros(num_samples)
        # order_log_prob_mask = tile2_logit.new_zeros(num_samples)

        order_score = order_logit + order_mask
        order_prob = F.softmax(order_score, dim=-1)
        order_density = Categorical(order_prob)
        order_action = order_density.sample()
        order_log_prob = order_density.log_prob(order_action)
        order_log_prob_mask = ((order_mask == 0).sum(dim=-1) > 1).float()

        log_probs = tile2_logit.new_zeros(num_samples, 1 + self.num_primes + 1)
        log_prob_masks = tile2_logit.new_zeros(num_samples, 1 + self.num_primes + 1)

        log_probs[:, 0] = order_log_prob
        log_prob_masks[:, 0] = order_log_prob_mask

        if cur_buffer_level == len(self.buffer_size_list):
            return (order_action, None, None), log_probs, log_prob_masks

        tile2_mask = tile_masks[:, loop_ind, 0]

        sp_tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            sp_tile2_mask_tmp[np.arange(num_samples), sp_tile2_max + i] = float('-inf')
        sp_tile2_mask_tmp = sp_tile2_mask_tmp[:, :tile2_mask.size(-1)]

        sp_tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), sp_tile2_mask_tmp], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            sp_tile2_mask_tmp[np.arange(num_samples), sp_tile2_min + tile2_mask.size(-1) - i] = float('-inf')
        sp_tile2_mask = sp_tile2_mask_tmp[:, tile2_mask.size(-1):]

        sp_tile2_score = sp_tile2_logit + sp_tile2_mask
        sp_tile2_prob = F.softmax(sp_tile2_score, dim=-1)
        sp_tile2_density = Categorical(sp_tile2_prob)
        sp_tile2_action = sp_tile2_density.sample()
        sp_tile2_log_prob = sp_tile2_density.log_prob(sp_tile2_action)
        sp_tile2_log_prob_mask = ((sp_tile2_mask == 0).sum(dim=-1) > 1).float()

        # if cur_buffer_level == 4:
        #     print(mode, sp_tile2_action[0], sp_tile2_max[0], tile2_max[0])

        tile2_min = sp_tile2_action
        tile2_max = torch.minimum(tile2_max, tile2_min + max_temporal_tile2.long())

        tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            tile2_mask_tmp[np.arange(num_samples), tile2_max + i] = float('-inf')
        tile2_mask_tmp = tile2_mask_tmp[:, :tile2_mask.size(-1)]

        tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), tile2_mask_tmp], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            tile2_mask_tmp[np.arange(num_samples), tile2_min + tile2_mask.size(-1) - i] = float('-inf')
        tile2_mask = tile2_mask_tmp[:, tile2_mask.size(-1):]

        # if self.pre_tile_budgets is not None and (cur_buffer_level == 2 or cur_buffer_level == 3):
        if self.pre_tile_budgets is not None and cur_buffer_level == len(self.buffer_size_list) - 1:
            # if cur_buffer_level == 2:
            #     loop_range = [0, 1, 2, 3, 4]
            #     # if loop_ind == 4:
            #     #     print(tile2_max[2], tile_remain_budgets[2, loop_ind, 0])
            # else:
            loop_range = [0, 1, 2, 3, 4, 5, 6, 7]

            if loop_ind in loop_range:
                tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
                for i in range(1, tile2_mask.size(-1) + 1):
                    tile2_mask_tmp[np.arange(num_samples), tile2_max + i] = float('-inf')
                tile2_mask_tmp = tile2_mask_tmp[:, :tile2_mask.size(-1)]

                tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), tile2_mask_tmp], dim=-1)
                for i in range(1, tile2_mask.size(-1)):
                    tile2_mask_tmp[np.arange(num_samples), tile2_max + tile2_mask.size(-1) - i] = float('-inf')
                tile2_mask = tile2_mask_tmp[:, tile2_mask.size(-1):]
            # if loop_ind == 5:
            #     print(cur_buffer_level, tile2_max, tile2_min, max_temporal_tile2.long(), tile2_remain_budgets)

        tile2_score = tile2_logit + tile2_mask
        tile2_prob = F.softmax(tile2_score, dim=-1)
        tile2_density = Categorical(tile2_prob)
        tile2_action = tile2_density.sample()
        tile2_log_prob = tile2_density.log_prob(tile2_action)
        tile2_log_prob_mask = ((tile2_mask == 0).sum(dim=-1) > 1).float()

        tile_action = tile2_action
        tile_actions = []
        sp_tile_actions = []
        log_probs = []
        log_prob_masks = []

        log_probs.append(order_log_prob)
        log_prob_masks.append(order_log_prob_mask)

        tile_actions.append(tile2_action)
        sp_tile_actions.append(sp_tile2_action)
        log_probs.append(tile2_log_prob)
        log_prob_masks.append(tile2_log_prob_mask)
        # if mode % self.steps_per_level in [4]:
        #     print("cur_buffer_level", cur_buffer_level, mode % self.steps_per_level, tile2_min, tile2_max,
        #           tile2_min + max_temporal_tile2.long(),
        #           tile_remain_budgets[:, loop_ind, 0])

        for p in range(1, self.num_primes):
            remain_buffer_size = remain_buffer_size / torch.pow(int(self.idx2prime[p - 1]), tile_action).float()
            tile_max = torch.log2(torch.clamp(remain_buffer_size, min=1)) / math.log2(int(self.idx2prime[p]))
            tile_max = torch.clamp(tile_max.long(), min=0, max=tile_masks.size(-1) - 1)
            tile_max = torch.minimum(tile_max, tile_remain_budgets[:, loop_ind, p])

            tile_mask = copy.deepcopy(tile_masks[:, loop_ind, p])
            tile_mask_tmp = torch.cat([tile_mask, torch.zeros_like(tile_mask)], dim=-1)
            for i in range(1, tile_mask.size(-1) + 1):
                tile_mask_tmp[np.arange(num_samples), tile_max + i] = float('-inf')
            tile_mask = tile_mask_tmp[:, :tile_mask.size(-1)]

            if self.pre_tile_budgets is not None and (cur_buffer_level == 2 or cur_buffer_level == 3):
                if cur_buffer_level == 2:
                    loop_range = [0, 1, 2, 3, 4]
                else:
                    loop_range = [0, 1, 2, 3, 4, 5, 6, 7]
                if loop_ind in loop_range:
                    tile_mask_tmp = torch.cat([tile_mask, torch.zeros_like(tile_mask)], dim=-1)
                    for i in range(1, tile_mask.size(-1) + 1):
                        tile_mask_tmp[np.arange(num_samples), tile_max + i] = float('-inf')
                    tile_mask_tmp = tile_mask_tmp[:, :tile_mask.size(-1)]

                    tile_mask_tmp = torch.cat([torch.zeros_like(tile_mask), tile_mask_tmp], dim=-1)
                    for i in range(1, tile_mask.size(-1)):
                        tile_mask_tmp[np.arange(num_samples), tile_max + tile_mask.size(-1) - i] = float('-inf')
                    tile_mask = tile_mask_tmp[:, tile_mask.size(-1):]

            tile_logit = tile_logits[:, p]
            tile_score = tile_logit + tile_mask
            tile_prob = F.softmax(tile_score, dim=-1)
            tile_density = Categorical(tile_prob)
            tile_action = tile_density.sample()
            tile_log_prob = tile_density.log_prob(tile_action)
            tile_log_prob_mask = ((tile_mask == 0).sum(dim=-1) > 1).float()
            tile_actions.append(tile_action)
            # if mode % self.steps_per_level == 4:
            #     print(cur_buffer_level, tile2_action)
            log_probs.append(tile_log_prob)
            log_prob_masks.append(tile_log_prob_mask)
            sp_tile_actions.append(program_seq.new_zeros(num_samples))

        log_probs.append(sp_tile2_log_prob)
        log_prob_masks.append(sp_tile2_log_prob_mask)

        tile_actions = torch.stack(tile_actions, dim=1)
        sp_tile_actions = torch.stack(sp_tile_actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        log_prob_masks = torch.stack(log_prob_masks, dim=1)

        return (order_action, tile_actions, sp_tile_actions), log_probs, log_prob_masks
