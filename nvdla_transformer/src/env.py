import numpy as np
import torch
import yaml
import os, sys
import copy
import random
from timeloop_env import Timeloop
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import shutil
import glob
import pickle
from datetime import datetime
import pandas as pd


class Environment(object):
    def __init__(self, pre_tile_budgets, accelerator, fitness_obj=['latency'], report_dir='./report',
                 use_pool=True, use_IO=True, log_level=0, save_chkpt=False, num_episodes=4):

        self.pre_tile_budgets = pre_tile_budgets

        self.fitness_obj = fitness_obj
        self.timeloop_out_config_path = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.report_dir = report_dir
        self.timeloop_env = Timeloop(in_config_path='../run_config',
                                        out_config_path=self.timeloop_out_config_path, accelerator=accelerator,
                                        opt_obj=self.fitness_obj)
        self.dim2note = self.timeloop_env.dim2note
        self.len_dimension = len(self.dim2note.values())
        self.num_buf_levels = self.timeloop_env.get_num_buffer_levels()
        # print(f'Number of buffer levels: {self.num_buf_levels}')
        self.buffer_size_list = self.timeloop_env.get_buffer_size_list()
        self.buf_spmap_cstr = self.timeloop_env.get_buffer_spmap_cstr()
        self.buffers_with_spmap = list(self.timeloop_env.get_buffers_with_spmap())
        self.dimension, self.dimension_prime, self.prime2idx = self.timeloop_env.get_dimension_primes()
        self.num_primes = len(self.prime2idx.keys())
        # print(self.buf_spmap_cstr, self.buffers_with_spmap, self.buffer_size_list, self.prime2idx)
        self.use_pool = bool(use_pool)
        self.use_IO = bool(use_IO)
        self.log_level = log_level
        self.idealperf = {}
        self.save_chkpt = save_chkpt

        self.best_fitness_record = []
        self.best_latency_record = []
        self.best_energy_record = []
        self.best_sol_record = []
        self.best_fitness = float("-Inf")
        self.best_latency = float("-Inf")
        self.best_energy = float("-Inf")
        self.best_sol = None

        self.set_dimension()
        # self.level_order = [1, 4,2,3, 5, 1]
        # self.level_order = [5, 2, 4, 1, 3, 6, 2]
        self.level_order = [2, 3, 1, 4, 1]
        self.start_level_order = 0
        self.cur_buffer_level = self.level_order[self.start_level_order]
        self.steps_per_level = len(self.dim2note.values())
        self.total_steps = self.num_buf_levels * self.steps_per_level
        self.mode = (self.cur_buffer_level - 1) * self.steps_per_level
        self.time_steps = 0
        self.num_episodes = num_episodes
        self.max_tile = 30
        self.min_reward = None

        self.initial_order_mask = np.zeros((self.num_episodes, self.len_dimension + 1), dtype=np.float)
        self.initial_order_mask[:, -1] = float('-inf')

        self.tile_budgets = np.zeros((self.num_episodes, self.len_dimension, self.num_primes), dtype=np.int32)
        self.initial_tile_masks = np.zeros(
            (self.num_episodes, self.len_dimension, self.num_primes, (self.max_tile + 1) * 2), dtype=np.float)

        # print("buffers_with_spmap", self.buffers_with_spmap)
        for i, key in enumerate(self.dim2note.values()):
            tile_budget = self.dimension_prime[key]
            for k, v in self.prime2idx.items():
                self.tile_budgets[:, i, v] = tile_budget[k]
                self.initial_tile_masks[:, i, v, tile_budget[k] + 1:] = float('-inf')

        length = self.num_primes * 2 + 1
        self.initial_program_seq = np.zeros((self.num_episodes, self.total_steps + 1, length), dtype=np.int32)
        self.initial_program_seq[:, 0, 0] = self.steps_per_level
        self.initial_program_seq[:, 0, 1: self.num_primes + 1] = self.max_tile
        self.initial_program_seq[:, 0, self.num_primes + 1: self.num_primes + 1 + self.num_primes] = self.max_tile

        for i in range(self.num_buf_levels):
            for j in range(self.steps_per_level):
                self.initial_program_seq[:, i * self.steps_per_level + j + 1, 0] = j
                self.initial_program_seq[:, i * self.steps_per_level + j + 1, self.num_primes + 1:] = 0
                if i == self.num_buf_levels - 1:
                    self.initial_program_seq[:, i * self.steps_per_level + j + 1,
                    1: self.num_primes + 1] = self.tile_budgets[:, j]
                else:
                    self.initial_program_seq[:, i * self.steps_per_level + j + 1, 1: self.num_primes + 1] = 0

        self.program_seq = copy.deepcopy(self.initial_program_seq[:, :1, :])
        self.program_seq_disorder = copy.deepcopy(self.initial_program_seq[:, 1:, :])
        self.final_program_seq = copy.deepcopy(self.initial_program_seq[:, 1:, :])

        self.tile_remain_budgets = copy.deepcopy(self.tile_budgets)

        # print("before: ", self.pre_tile_budgets)
        if self.pre_tile_budgets is not None:
            self.pre_tile_budgets[0] = self.tile_remain_budgets[0, 0]
            self.pre_tile_budgets[1] = self.tile_remain_budgets[0, 1]
            self.pre_tile_budgets[2] = np.minimum(self.tile_remain_budgets[0, 2], self.pre_tile_budgets[2])
            self.pre_tile_budgets[3] = np.minimum(self.tile_remain_budgets[0, 3], self.pre_tile_budgets[3])
            if self.pre_tile_budgets[4] is not None:
                self.pre_tile_budgets[4] = np.minimum(self.tile_remain_budgets[0, 4], self.pre_tile_budgets[4])
            else:
                self.pre_tile_budgets[4] = self.tile_remain_budgets[0, 4]
            self.pre_tile_budgets[5] = np.minimum(self.tile_remain_budgets[0, 5], self.pre_tile_budgets[5])
            self.pre_tile_budgets[6] = np.minimum(self.tile_remain_budgets[0, 6], self.pre_tile_budgets[6])
            self.pre_tile_budgets[7] = np.minimum(self.tile_remain_budgets[0, 7], self.pre_tile_budgets[7])
            # print("after ", self.pre_tile_budgets)

            self.pre_tile_budgets = np.array(self.pre_tile_budgets)
            self.tile_remain_budgets[:, 0:] = self.pre_tile_budgets[0:]

        # print(self.pre_tile_budgets)

        self.tile_masks = copy.deepcopy(self.initial_tile_masks)
        self.order_mask = copy.deepcopy(self.initial_order_mask)

    # def get_default_density(self):
    #     density = {'Weights': 1,
    #                'Inputs': 1,
    #                'Outputs': 1}
    #     return density

    def set_dimension(self):
        # self.idealperf['edp'], self.idealperf['latency'], self.idealperf['energy'] = self.timeloop_env.get_ideal_perf(self.dimension)
        # self.idealperf['utilization'] = 1

        self.best_fitness_record = []
        self.best_latency_record = []
        self.best_energy_record = []
        self.best_sol_record = []
        self.best_fitness = float("-Inf")
        self.best_latency = float("-Inf")
        self.best_energy = float("-Inf")
        self.best_sol = None

    def reset(self):
        self.start_level_order = 0
        self.cur_buffer_level = self.level_order[self.start_level_order]
        self.mode = (self.cur_buffer_level - 1) * self.steps_per_level
        self.time_steps = 0

        self.program_seq = copy.deepcopy(self.initial_program_seq[:, :1, :])
        self.program_seq_disorder = copy.deepcopy(self.initial_program_seq[:, 1:, :])
        self.final_program_seq = copy.deepcopy(self.initial_program_seq[:, 1:, :])
        self.final_program_seq = copy.deepcopy(self.initial_program_seq[:, 1:, :])

        self.tile_remain_budgets = copy.deepcopy(self.tile_budgets)

        if self.pre_tile_budgets is not None:
            self.tile_remain_budgets[:, 0:] = self.pre_tile_budgets[0:]

        self.tile_masks = copy.deepcopy(self.initial_tile_masks)
        self.order_mask = copy.deepcopy(self.initial_order_mask)
        return (self.program_seq, self.order_mask, self.tile_remain_budgets, self.tile_masks[:, :, :, 0:self.max_tile + 1],
                self.mode, self.cur_buffer_level, self.program_seq_disorder)

    def get_reward(self, final_trg_seq):
        if self.use_pool:
            pool = ProcessPoolExecutor(self.num_episodes)
            self.timeloop_env.create_pool_env(num_pools=self.num_episodes)
        else:
            pool = None
            self.timeloop_env.create_pool_env(num_pools=1)

        reward = self.evaluate(final_trg_seq[:, :, :], pool)

        return reward

    def step(self, actions):
        loop_ind = self.mode % self.steps_per_level

        if self.cur_buffer_level < self.num_buf_levels:
            step_order, step_tile, step_parallel = actions
            step_order = step_order.cpu().numpy()
            step_tile = step_tile.cpu().numpy()
            step_parallel = step_parallel.cpu().numpy()
            length = self.num_primes * 2 + 1
            cur_seq = np.zeros((self.num_episodes, length), dtype=np.int32)
            cur_seq[:, 0] = step_order
            cur_seq[:, 1: self.num_primes + 1] = step_tile
            cur_seq[:, self.num_primes + 1:] = step_parallel

            self.program_seq = np.concatenate((self.program_seq, np.expand_dims(cur_seq, 1)), axis=1)

            seq_disorder_ind = (self.cur_buffer_level - 1) * self.steps_per_level + loop_ind
            self.program_seq_disorder[:, seq_disorder_ind, 1: self.num_primes + 1] = step_tile
            self.program_seq_disorder[:, seq_disorder_ind, self.num_primes + 1:] = step_parallel

            self.final_program_seq[:, self.mode, 0] = step_order
            self.final_program_seq[:, self.mode, 1: self.num_primes + 1] = step_tile
            self.final_program_seq[:, self.mode, self.num_primes + 1:] = step_parallel

            self.order_mask[torch.arange(self.num_episodes), step_order] = float('-inf')
            self.tile_remain_budgets[:, loop_ind] -= step_tile

            for i in range(1, self.max_tile + 1):
                for j in range(0, self.num_primes):
                    tile_remain_budget = self.tile_remain_budgets[:, loop_ind, j]
                    self.tile_masks[np.arange(self.num_episodes), loop_ind, j, tile_remain_budget + i] = float('-inf')

        else:
            step_order, _, _ = actions
            step_order = step_order.cpu().numpy()
            step_parallel = np.zeros((self.num_episodes, 1), dtype=np.int32)

            if self.pre_tile_budgets is not None:
                tile_remain_budgets2 = np.zeros((self.num_episodes, self.len_dimension, self.num_primes), dtype=np.int32)
                # tile_remain_budgets2[:, 0, :] = 0
                # tile_remain_budgets2[:, 1, :] = 0
                # tile_remain_budgets2[:, 2] = self.tile_budgets[:, 2] - self.pre_tile_budgets[2]
                # tile_remain_budgets2[:, 3] = self.tile_budgets[:, 3] - self.pre_tile_budgets[3]
                # tile_remain_budgets2[:, 4] = self.tile_budgets[:, 4] - self.pre_tile_budgets[4]
                # tile_remain_budgets2[:, 5] = self.tile_budgets[:, 5] - self.pre_tile_budgets[5]
                # tile_remain_budgets2[:, 6] = self.tile_budgets[:, 6] - self.pre_tile_budgets[6]
                # tile_remain_budgets2[:, 7] = self.tile_budgets[:, 7] - self.pre_tile_budgets[7]
                tile_remain_budgets2[:, 0:] = self.tile_budgets[:, 0:] - self.pre_tile_budgets[0:]
            else:
                tile_remain_budgets2 = copy.deepcopy(self.tile_remain_budgets)
            # print(self.tile_budgets[0, 0:], self.tile_remain_budgets[0], self.pre_tile_budgets[0:], tile_remain_budgets2[0])
            step_tile = tile_remain_budgets2[:, loop_ind]

            length = self.num_primes * 2 + 1
            cur_seq = np.zeros((self.num_episodes, length), dtype=np.int32)
            cur_seq[:, 0] = step_order
            cur_seq[:, 1: self.num_primes + 1] = step_tile
            cur_seq[:, self.num_primes + 1:] = step_parallel

            self.program_seq = np.concatenate((self.program_seq, np.expand_dims(cur_seq, 1)), axis=1)

            seq_disorder_ind = (self.cur_buffer_level - 1) * self.steps_per_level + loop_ind
            self.program_seq_disorder[:, seq_disorder_ind, 1: self.num_primes + 1] = step_tile
            self.program_seq_disorder[:, seq_disorder_ind, self.num_primes + 1:] = step_parallel

            self.final_program_seq[:, self.mode, 0] = step_order
            self.final_program_seq[:, self.mode, 1: self.num_primes + 1] = step_tile
            self.final_program_seq[:, self.mode, self.num_primes + 1:] = step_parallel

            self.order_mask[torch.arange(self.num_episodes), step_order] = float('-inf')

        self.time_steps += 1
        self.mode += 1
        if self.mode % self.steps_per_level == 0:
            self.start_level_order += 1
            self.cur_buffer_level = self.level_order[self.start_level_order]
            self.mode = (self.cur_buffer_level - 1) * self.steps_per_level
            self.order_mask = copy.deepcopy(self.initial_order_mask)

        # order_mask = copy.deepcopy(self.order_mask)
        # tile_remain_budgets = copy.deepcopy(self.tile_remain_budgets)
        # tile_masks = copy.deepcopy(self.tile_masks[:, :, :, 0:self.max_tile + 1])
        #
        # pro = copy.deepcopy(self.trg_seq)
        # trg_seq_disorder = copy.deepcopy(self.trg_seq_disorder)

        if self.time_steps < self.total_steps:
            done = 0
            info = None
            reward = np.zeros(self.num_episodes)
        else:
            done = 1
            info = None
            reward_saved = copy.deepcopy(self.get_reward(self.final_program_seq))
            # reward_saved[reward_saved==float('-inf')] = self.min_reward
            sort_idx = np.argsort(reward_saved)
            top_k_idx = sort_idx[int(self.num_episodes / 4) - 1]
            reward = (reward_saved - reward_saved[top_k_idx])
            # print(reward_saved, reward_saved[top_k_idx], reward)
            # reward_saved[reward_saved==float('-inf')] = float('inf')
            # if self.min_reward is None:
            #     self.min_reward = reward_saved.min()
            # else:
            #     self.min_reward = min(self.min_reward, reward_saved.min())
            # reward_saved[reward_saved == float('inf')] = self.min_reward
            # reward = reward_saved - self.min_reward
            # reward_saved[reward_saved==float('inf')] = reward_saved.min()
            # reward = (reward_saved - reward_saved.min()) / (reward_saved.std() + 1e-12)
            # self.last_reward = reward_saved

        return (self.program_seq, self.order_mask, self.tile_remain_budgets, self.tile_masks[:, :, :, 0:self.max_tile + 1],
                self.mode, self.cur_buffer_level, self.program_seq_disorder), reward, done, info

    def thread_fun(self, args, fitness_obj=None):
        sol, pool_idx = args
        fit = self.timeloop_env.run_timeloop(self.dimension, sol, pool_idx=pool_idx, use_IO=self.use_IO,
                                             fitness_obj=fitness_obj if fitness_obj is not None else self.fitness_obj)
        return fit

    def evaluate(self, sols, pool):
        fitness = np.ones((self.num_episodes, len(self.fitness_obj))) * np.NINF
        if not pool:
            for i, sol in enumerate(sols):
                fit = self.thread_fun((sol, 0))
                fitness[i] = fit
        else:
            while(1):
                try:
                    fits = list(pool.map(self.thread_fun, zip(sols, np.arange(len(sols)))))
                    for i, fit in enumerate(fits):
                        fitness[i] = fit
                    break
                except Exception as e:
                    if self.log_level>2:
                        print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(self.num_episodes)

        latency_fitness = fitness[:, 1]
        energy_fitness = fitness[:, 2]
        fitness = fitness[:, 0]
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_latency = latency_fitness[best_idx]
            self.best_energy = energy_fitness[best_idx]
            self.best_sol = sols[best_idx]
            self.create_timeloop_report(self.best_sol, self.report_dir)
        # print("Achieved Fitness: ", self.best_fitness, self.mode, fitness, self.best_latency, self.best_energy, (fitness > float('-inf')).sum())
        return fitness

    def create_timeloop_report(self, sol, dir_path):
        fitness = self.thread_fun((sol, 0))
        stats = self.thread_fun((sol, 0), fitness_obj='all')
        os.makedirs(dir_path, exist_ok=True)
        columns = ['EDP (uJ cycles)', 'Cycles', 'Energy (pJ)', 'Utilization', 'pJ/Algorithm-Compute', 'pJ/Actual-Compute', 'Area (mm2)'][:len(stats)]
        os.system(f'cp -d -r {os.path.join(self.timeloop_out_config_path, "pool-0")}/* {dir_path}')
        with open(os.path.join(dir_path,'RL-Timeloop.txt'), 'w') as fd:
            value = [f'{v:.5e}' for v in fitness]
            fd.write(f'Achieved Fitness: {value}\n')
            fd.write(f'Statistics\n')
            fd.write(f'{columns}\n')
            fd.write(f'{stats}')
        stats = np.array(stats).reshape(1, -1)
        df = pd.DataFrame(stats, columns=columns)
        df.to_csv(os.path.join(dir_path,'Timeloop.csv'))

    def record_chkpt(self, write=False):
        self.best_fitness_record.append(self.best_fitness)
        self.best_latency_record.append(self.best_latency)
        self.best_energy_record.append(self.best_energy)
        self.best_sol_record.append(self.best_sol)

        chkpt = None
        if write:
            with open(os.path.join(self.report_dir, 'env_chkpt.plt'), 'wb') as fd:
                chkpt = {
                    'best_fitness_record': self.best_fitness_record,
                    'best_latency_record': self.best_latency_record,
                    'best_energy_record': self.best_energy_record,
                    'best_sol_record': self.best_sol_record,
                    'best_fitness': self.best_fitness,
                    'best_latency': self.best_latency,
                    'best_energy': self.best_energy,
                    'best_sol': self.best_sol
                }
                pickle.dump(chkpt, fd)
        return chkpt

    def clean_timeloop_output_files(self):
        shutil.rmtree(self.timeloop_out_config_path)
        out_prefix = "./timeloop-model."
        output_file_names = []
        output_file_names.append( "tmp-accelergy.yaml")
        output_file_names.append(out_prefix + "accelergy.log")
        output_file_names.extend(glob.glob("*accelergy.log"))
        output_file_names.extend(glob.glob("*tmp-accelergy.yaml"))
        output_file_names.append(out_prefix + ".log")
        output_file_names.append(out_prefix + "ART.yaml")
        output_file_names.append(out_prefix + "ART_summary.yaml")
        output_file_names.append(out_prefix + "ERT.yaml")
        output_file_names.append(out_prefix + "ERT_summary.yaml")
        output_file_names.append(out_prefix + "flattened_architecture.yaml")
        output_file_names.append(out_prefix + "map+stats.xml")
        output_file_names.append(out_prefix + "map.txt")
        output_file_names.append(out_prefix + "stats.txt")
        for f in output_file_names:
            if os.path.exists(f):
                os.remove(f)








