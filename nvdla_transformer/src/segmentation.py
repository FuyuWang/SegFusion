import os
import yaml
import re

import math
import numpy as np
import random
import copy
from datetime import datetime
from collections import defaultdict, OrderedDict
from intra_op import intra_op_mapping_dse


class Node:
    def __init__(self, index, name, operator):
        self.index = index
        self.name = name
        self.operator = operator
        self.pred_nodes = []
        self.succ_nodes = []
        self.active = False
        self.active_nodes = []
        self.pre_tile_budgets = None

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False


class Graph:
    def __init__(self):
        self.nodes = {}

    def activate_node(self, node_index):
        self.nodes[node_index].activate()

    def deactivate_node(self, node_index):
        self.nodes[node_index].deactivate()


class Workload:
    def __init__(self, workload_dir, workload_name, batch_size, layer2budgets):
        self.workload_dir = workload_dir
        self.workload_name = workload_name
        self.batch_size = batch_size
        self.prime2idx = None
        self.layer2budgets = layer2budgets

        # self.primes = [2, 3, 5, 7]
        # if workload_name in ['resnet50', 'mobilenetv2']:

        with open(os.path.join(workload_dir, '{}_problems/layers.yaml'.format(workload_name)), 'r') as fd:
            self.layers = yaml.load(fd, Loader=yaml.SafeLoader)
        fd.close()
        self.graph = Graph()

        for layer_id, layer_name in enumerate(self.layers):
            with open(os.path.join(workload_dir, '{}_problems/{}.yaml'.format(workload_name, layer_name)),
                      'r') as fd:
                operator = yaml.load(fd, Loader=yaml.SafeLoader)
            fd.close()

            if 'H' not in operator['problem'].keys():
                operator['problem']['H'] = 1

            if 'type' in operator['problem'].keys() and operator['problem']['type'] == 'BMM':
                operator['problem']['H'] = operator['problem']['H'] * batch_size
            else:
                operator['problem']['N'] = operator['problem']['N'] * batch_size

            if 'type' not in operator['problem'].keys():
                operator['problem']['type'] = 'C2D'

            node = Node(layer_id, layer_name, operator)

            dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}
            dimension_dict = {}
            for key in dim2note.values():
                value = operator['problem'][key]
                dimension_dict[key] = value
            dimension_prime = {key: get_prime_factors(dimension_dict[key]) for key in dim2note.values()}
            primes = set()
            for i, key in enumerate(dim2note.values()):
                tile_budget = dimension_prime[key]
                for k in tile_budget.keys():
                    primes.add(int(k))
            primes = sorted(primes)
            prime2idx = {'{}'.format(pf): i for i, pf in enumerate(primes)}
            self.prime2idx = prime2idx
            num_primes = len(prime2idx.keys())

            tile_budgets = np.zeros((len(dim2note.values()), num_primes), dtype=np.int32)
            for i, key in enumerate(dim2note.values()):
                tile_budget = dimension_prime[key]
                for k, v in prime2idx.items():
                    tile_budgets[i, v] = tile_budget[k]

            R = operator['problem']['R']
            S = operator['problem']['S']
            C = operator['problem']['C']
            wstride = operator['problem']['Wstride']
            hstride = operator['problem']['Hstride']
            wdilation = operator['problem']['Wdilation']
            hdilation = operator['problem']['Hdilation']

            P_budgets, Q_budgets, C_budgets, K_budgets, H_budgets, N_budgets = layer2budgets[layer_id]

            tile_budgets[2] = np.minimum(tile_budgets[2], P_budgets)
            tile_budgets[3] = np.minimum(tile_budgets[3], Q_budgets)
            tile_budgets[4] = np.minimum(tile_budgets[4], C_budgets)
            tile_budgets[5] = np.minimum(tile_budgets[5], K_budgets)
            tile_budgets[6] = np.minimum(tile_budgets[6], H_budgets)
            tile_budgets[7] = np.minimum(tile_budgets[7], N_budgets)

            node.pre_tile_budgets = tile_budgets

            if layer_id > 0:
                node.pred_nodes.append(layer_id - 1)
            if layer_id < len(self.layers) - 1:
                node.succ_nodes.append(layer_id + 1)
            self.graph.nodes[layer_id] = node
        # if workload_name == 'resnet50':
        #     self.graph.nodes[0].succ_nodes = [1, 2]
        #     self.graph.nodes[1].succ_nodes = [5]
        #     self.graph.nodes[2].pred_nodes = [0]
        #     self.graph.nodes[5].pred_nodes = [1, 4]
        #     self.graph.nodes[2].active_nodes = [1]
        #     self.graph.nodes[3].active_nodes = [1]
        #     self.graph.nodes[4].active_nodes = [1]
        #
        #     self.graph.nodes[10].succ_nodes = [11, 12]
        #     self.graph.nodes[11].succ_nodes = [15]
        #     self.graph.nodes[12].pred_nodes = [10]
        #     self.graph.nodes[15].pred_nodes = [11, 14]
        #     self.graph.nodes[12].active_nodes = [11]
        #     self.graph.nodes[13].active_nodes = [11]
        #     self.graph.nodes[14].active_nodes = [11]
        #
        #     self.graph.nodes[23].succ_nodes = [24, 25]
        #     self.graph.nodes[24].succ_nodes = [28]
        #     self.graph.nodes[25].pred_nodes = [23]
        #     self.graph.nodes[28].pred_nodes = [24, 27]
        #     self.graph.nodes[25].active_nodes = [24]
        #     self.graph.nodes[26].active_nodes = [24]
        #     self.graph.nodes[27].active_nodes = [24]
        #
        #     self.graph.nodes[42].succ_nodes = [43, 44]
        #     self.graph.nodes[43].succ_nodes = [47]
        #     self.graph.nodes[44].pred_nodes = [42]
        #     self.graph.nodes[47].pred_nodes = [43, 46]
        #     self.graph.nodes[44].active_nodes = [43]
        #     self.graph.nodes[45].active_nodes = [43]
        #     self.graph.nodes[46].active_nodes = [43]
        print(self.graph.nodes.keys())

    def node_input_weight_size(self, node):

        R = node.operator['problem']['R']
        S = node.operator['problem']['S']
        C = node.operator['problem']['C']
        wstride = node.operator['problem']['Wstride']
        hstride = node.operator['problem']['Hstride']
        wdilation = node.operator['problem']['Wdilation']
        hdilation = node.operator['problem']['Hdilation']
        P = 1
        Q = 1
        C_tile_size = 1
        K = 1
        H = 1
        N = 1

        node_tile_budgets = node.pre_tile_budgets

        if len(node.pred_nodes) > 0:
            lcm_tile_budgets = copy.deepcopy(node_tile_budgets)
            for pred_node in node.pred_nodes:
                pred_node_tile_budgets = self.graph.nodes[pred_node].pre_tile_budgets
                lcm_tile_budgets[2:4] = np.maximum(pred_node_tile_budgets[2:4], node_tile_budgets[2:4])
                lcm_tile_budgets[7] = np.maximum(pred_node_tile_budgets[7], node_tile_budgets[7])

                if node.operator['problem']['type'] == 'BMM' and self.graph.nodes[pred_node].operator['problem']['type'] == 'C2D':
                    lcm_tile_budgets[6] = np.maximum(pred_node_tile_budgets[5], node_tile_budgets[6])
                elif node.operator['problem']['type'] == 'C2D' and self.graph.nodes[pred_node].operator['problem']['type'] == 'BMM':
                    lcm_tile_budgets[4] = np.maximum(pred_node_tile_budgets[5] + pred_node_tile_budgets[6], node_tile_budgets[4])
                else:
                    lcm_tile_budgets[4] = np.maximum(pred_node_tile_budgets[5], node_tile_budgets[4])
                    lcm_tile_budgets[6] = np.maximum(pred_node_tile_budgets[6], node_tile_budgets[6])
                print(lcm_tile_budgets, node.operator['problem']['H'], self.graph.nodes[pred_node].operator['problem']['H'])
            for k, v in self.prime2idx.items():
                P *= math.pow(int(k), lcm_tile_budgets[2, v])
                Q *= math.pow(int(k), lcm_tile_budgets[3, v])
                C_tile_size *= math.pow(int(k), lcm_tile_budgets[4, v])
                K *= math.pow(int(k), node_tile_budgets[5, v])  # outsize dim
                H *= math.pow(int(k), lcm_tile_budgets[6, v])
                N *= math.pow(int(k), lcm_tile_budgets[7, v])
        else:
            for k, v in self.prime2idx.items():
                P *= math.pow(int(k), node_tile_budgets[2, v])
                Q *= math.pow(int(k), node_tile_budgets[3, v])
                C_tile_size *= math.pow(int(k), node_tile_budgets[4, v])
                K *= math.pow(int(k), node_tile_budgets[5, v])  # outsize dim
                H *= math.pow(int(k), node_tile_budgets[6, v])
                N *= math.pow(int(k), node_tile_budgets[7, v])

        # if node.operator['problem']['type'] == 'T2D':
        #     input_size = P * Q * C * N * H
        #     input_tiling_C_size = P * Q * C_tile_size * N * H
        #     # output_size = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
        #     #         (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
        # else:
        #     input_size = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
        #             (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C * H
        #     input_tiling_C_size = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
        #             (Q - 1) * hstride + 1 + hdilation * (S - 1)) * C_tile_size * H
        #     # output_size = N * P * Q * K * H
        #
        # weight_size = K * C * R * S * H
        # weight_tiling_C_size = K * C_tile_size * R * S * H

        input_size = N * C * H
        input_tiling_C_size = N * C_tile_size * H

        weight_size = K * C * H
        weight_tiling_C_size = K * C_tile_size * H

        node_active_output_size = 0
        if len(node.active_nodes) > 0:
            for active_node_idx in node.active_nodes:
                active_node = self.graph.nodes[active_node_idx]
                active_node_tile_budgets = active_node.pre_tile_budgets
                R = active_node.operator['problem']['R']
                S = active_node.operator['problem']['S']
                wstride = active_node.operator['problem']['Wstride']
                hstride = active_node.operator['problem']['Hstride']
                wdilation = active_node.operator['problem']['Wdilation']
                hdilation = active_node.operator['problem']['Hdilation']
                P = 1
                Q = 1
                K = 1
                H = 1
                N = 1
                for k, v in self.prime2idx.items():
                    P *= math.pow(int(k), active_node_tile_budgets[2, v])
                    Q *= math.pow(int(k), active_node_tile_budgets[3, v])
                    K *= math.pow(int(k), active_node_tile_budgets[5, v])
                    H *= math.pow(int(k), active_node_tile_budgets[6, v])
                    N *= math.pow(int(k), active_node_tile_budgets[7, v])
                # if active_node.operator['problem']['type'] == 'T2D':
                #     output_size = N * ((P - 1) * wstride + 1 + wdilation * (R - 1)) * (
                #             (Q - 1) * hstride + 1 + hdilation * (S - 1)) * K * H
                # else:
                #     output_size = N * P * Q * K * H
                output_size = N * P * Q * K * H
                node_active_output_size += output_size

        return weight_size, input_size, weight_tiling_C_size, input_tiling_C_size, node_active_output_size

    def pred_nodes_visited(self, node, visited_nodes):
        flag = True
        if len(node.pred_nodes) > 0:
            for pred_node in node.pred_nodes:
                if pred_node not in visited_nodes:
                    flag = False
                    break
        return flag

    def greedy_segmentation(self, shared_buffer_size):

        segmentation_sol = [0]
        input_sizes = []
        weight_sizes = []
        # cur_level_tile_budgets = None
        # last_level_tile_budgets = None
        # last_level_problem = None
        visit_nodes = []
        for idx, node in self.graph.nodes.items():
            visit_nodes.append(idx)
            node_weight_size, node_input_size, node_weight_tiling_C_size, node_input_tiling_C_size, node_active_output_size \
                = self.node_input_weight_size(node)

            succ_node_weight_size, succ_node_input_size = 0, 0
            if len(node.succ_nodes) > 0:
                for succ_node in node.succ_nodes:
                    if self.pred_nodes_visited(self.graph.nodes[succ_node], visit_nodes):
                        (succ_node_weight_size0, succ_node_input_size0,
                         succ_node_weight_tiling_C_size0, succ_node_input_tiling_C_size0, _) \
                            = self.node_input_weight_size(self.graph.nodes[succ_node])
                        # succ_node_weight_size += succ_node_weight_size0
                        # succ_node_input_size = max(succ_node_input_size, succ_node_input_size0)
                        # succ_node_input_size += succ_node_input_size0
                        succ_node_weight_size += succ_node_weight_tiling_C_size0
                        succ_node_input_size += succ_node_input_tiling_C_size0

            # if segmentation_sol[-1] == idx:
            #     node_weight_tiling_C_size = node_weight_size
            #     node_input_tiling_C_size = node_input_size

            succ_total_weight_size = np.array(weight_sizes).sum() + node_weight_tiling_C_size + succ_node_weight_size
            succ_total_input_size = copy.deepcopy(input_sizes)
            succ_total_input_size.append(node_input_tiling_C_size + node_active_output_size)
            succ_total_input_size.append(succ_node_input_size)
            succ_total_input_size = np.array(succ_total_input_size)
            succ_total_input_size[1:] += succ_total_input_size[0:-1]
            succ_used_shared_buffer_size = succ_total_weight_size + succ_total_input_size.max()
            print(idx, node_active_output_size, weight_sizes, input_sizes,
                  node_weight_tiling_C_size, node_input_tiling_C_size,
                  succ_node_weight_size, succ_node_input_size,
                  succ_total_weight_size, succ_total_input_size.max())

            if succ_used_shared_buffer_size <= shared_buffer_size:
                weight_sizes.append(node_weight_tiling_C_size)
                input_sizes.append(node_input_tiling_C_size)
            else:
                input_sizes = []
                weight_sizes = []
                # cur_level_tile_budgets = None
                # last_level_tile_budgets = None
                # total_weight_size = np.array(weight_sizes).sum() + cur_level_weight_size
                # total_input_size = copy.deepcopy(input_sizes)
                # total_input_size.append(cur_level_input_size)
                # total_input_size = np.array(total_input_size)
                # total_input_size[1:] += total_input_size[0:-1]
                # cur_used_shared_buffer_size = total_weight_size + total_input_size.max()
                # if cur_used_shared_buffer_size <= shared_buffer_size:
                segmentation_sol.append(idx + 1)

        if segmentation_sol[-1] != len(self.graph.nodes.items()):
            segmentation_sol.append(len(self.graph.nodes.items()))
        # print(gb_tile_budgets_sol)
        return segmentation_sol

    def segment_perf(self, accelerator, architecture, save_dir, segment_start, segment_end):
        problem_dir = '../../benchmarks/'

        segment_latency = []
        segment_energy = []
        segment_shared_buffer = []
        total_dram_accesses = 0
        total_operations = 0
        for level_id in range(segment_start, segment_end):
            print(level_id, self.layers[level_id])
            report_dir = os.path.join(save_dir, 'level-{}{}'.format(level_id, self.layers[level_id]))
            level_shared_buffer_size = np.array([0., 0.])
            level_latency = 0
            level_energy = 0

            operator = self.graph.nodes[level_id].operator
            problem = {'problem': {
                'shape': {'name': 'CNN-Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                          'coefficients': [{'name': 'Wstride', 'default': 1},
                                           {'name': 'Hstride', 'default': 1},
                                           {'name': 'Wdilation', 'default': 1},
                                           {'name': 'Hdilation', 'default': 1}],
                          },
                'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                             'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                             }}}

            problem['problem']['shape']['data-spaces'] = [
                {'name': 'Weights',
                 'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                  [['R', 'Wdilation'],
                                                   ['P', 'Wstride']],
                                                  [['S', 'Hdilation'],
                                                   ['Q', 'Hstride']]]},
                {'name': 'Outputs',
                 'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                 'read-write': True}]
            problem['problem']['instance']['type'] = 'C2D'
            print(operator)
            problem['problem']['instance']['N'] = operator['problem']['N']
            problem['problem']['instance']['H'] = operator['problem']['H']
            problem['problem']['instance']['K'] = operator['problem']['K']
            problem['problem']['instance']['C'] = operator['problem']['C']
            problem['problem']['instance']['P'] = operator['problem']['P']
            problem['problem']['instance']['Q'] = operator['problem']['Q']
            problem['problem']['instance']['R'] = operator['problem']['R']
            problem['problem']['instance']['S'] = operator['problem']['S']
            problem['problem']['instance']['Wstride'] = operator['problem']['Wstride']
            problem['problem']['instance']['Hstride'] = operator['problem']['Hstride']
            problem['problem']['instance']['Wdilation'] = operator['problem']['Wdilation']
            problem['problem']['instance']['Hdilation'] = operator['problem']['Hdilation']
            total_operations += problem['problem']['instance']['N'] * problem['problem']['instance']['H'] * \
                                problem['problem']['instance']['K'] * problem['problem']['instance']['C']

            if level_id == segment_start:
                shared_buffer_size = config_arch_mapspace(accelerator=accelerator, arch_file=architecture,
                                                          first_layer=True)
            else:
                shared_buffer_size = config_arch_mapspace(accelerator=accelerator, arch_file=architecture,
                                                          first_layer=False)
            with open('../run_config/{}/problem.yaml'.format(accelerator), 'w') as fd:
                yaml.dump(problem, fd)
            fd.close()

            if segment_start == segment_end - 1:
                pre_tile_budgets = None
            else:
                pre_tile_budgets = [None, None, self.layer2budgets[level_id][0],
                                    self.layer2budgets[level_id][1],
                                    self.layer2budgets[level_id][2],
                                    self.layer2budgets[level_id][3],
                                    self.layer2budgets[level_id][4],
                                    self.layer2budgets[level_id][5]]
                # if level_id == segment_end - 1 or level_id == segment_start:
                #     pre_tile_budgets[4] = None
            layer_chkpt = intra_op_mapping_dse(accelerator, problem, report_dir, pre_tile_budgets)
            # print(layer_chkpt['best_latency'], layer_chkpt['best_energy'])

            with open(os.path.join(report_dir, 'timeloop-model.stats.txt'), 'r') as f:
                lines = f.readlines()
            memory_latency = []
            memory_energy = []
            memory_bandwidth = []
            smem_line_id = None
            dram_line_id = None
            tmp_energy = None
            for li, line in enumerate(lines):
                m = re.match(r"Energy: (.*) uJ", line)
                if m:
                    tmp_energy = m.group(1)

                m1 = re.match(r"    Cycles               : (.*)", line)
                m2 = re.match(r"    Cycles                  : (.*)", line)
                if m1:
                    memory_latency.append(float(m1.group(1)))
                if m2:
                    memory_latency.append(float(m2.group(1)))

                m1 = re.match(r"        Energy \(total\)                           : (.*) pJ", line)
                m2 = re.match(r"    Energy \(total\)          : (.*) pJ", line)
                if m1:
                    memory_energy.append(float(m1.group(1)))
                if m2:
                    memory_energy.append(float(m2.group(1)))

                m = re.match(r"        Shared Bandwidth \(total\)                 : (.*) words/cycle", line)
                if m:
                    memory_bandwidth.append(float(m.group(1)))

                m = re.match(r"Level 3", line)
                if m:
                    smem_line_id = li
                m = re.match(r"Level 4", line)
                if m:
                    dram_line_id = li
            f.close()

            if level_id == segment_end - 1:
                level_latency += -1 * layer_chkpt['best_latency']
                level_energy += -1 * layer_chkpt['best_energy']
                # print("memory_latency: ", memory_latency, "memory_energy: ", memory_energy)
                dram_used_bandwidth = memory_bandwidth[4:]
                num_dram_accesses = np.array(dram_used_bandwidth) * memory_latency[-1]
                print("weight_access: ", num_dram_accesses[0], "output_access: ", num_dram_accesses[-1])
                total_dram_accesses += num_dram_accesses[0]
                total_dram_accesses += num_dram_accesses[-1]
            else:
                smem_energy_ratio = min(memory_energy[3] / (memory_latency[3] * memory_bandwidth[2] + 1e-6),
                                        memory_energy[4] / (memory_latency[3] * memory_bandwidth[3] + 1e-6))

                dram_energy_ratio = min(memory_energy[5] / (memory_latency[4] * memory_bandwidth[4] + 1e-6),
                                        memory_energy[-1] / (memory_latency[4] * memory_bandwidth[-1] + 1e-6))

                memory_energy[-1] /= (dram_energy_ratio / smem_energy_ratio)
                print("memory_latency: ", memory_latency)
                print("memory_energy: ", memory_energy, np.array(memory_energy).sum() * 1e-6, tmp_energy)
                dram_used_bandwidth = memory_bandwidth[4:]

                sub_seme_bandwidth = 256 * 256
                num_dram_accesses = np.array(dram_used_bandwidth) * memory_latency[-1]
                dram_bandwidth = np.array(dram_used_bandwidth).sum()
                if len(dram_used_bandwidth) == 2:
                    dram_weight_latency = num_dram_accesses[0] / dram_bandwidth
                    dram_output_latency = num_dram_accesses[-1] / sub_seme_bandwidth
                    dram_latency = max(dram_weight_latency, dram_output_latency)
                else:
                    dram_output_latency = num_dram_accesses[-1] / sub_seme_bandwidth
                    dram_latency = max(dram_output_latency,
                                       (num_dram_accesses[0] + num_dram_accesses[1]) / dram_bandwidth)
                    total_dram_accesses += num_dram_accesses[0]
                    total_dram_accesses += num_dram_accesses[1]
                    print("weight_access: ", num_dram_accesses[0], "input_access: ", num_dram_accesses[1])
                level_latency += max(np.array(memory_latency[:-1]).max(), dram_latency)
                print("latency comparison: ", np.array(memory_latency).max(),
                      max(np.array(memory_latency[:-1]).max(), dram_latency))

            layer_shared_buffer_size = []
            for li in range(smem_line_id, dram_line_id):
                line = lines[li]
                m = re.match(r"        Utilized capacity                        : (.*)", line)
                if m:
                    layer_shared_buffer_size.append(float(m.group(1)))
            layer_shared_buffer_size = np.array(layer_shared_buffer_size)
            level_shared_buffer_size += layer_shared_buffer_size

            segment_latency.append(level_latency)
            segment_energy.append(level_energy)

            segment_shared_buffer.append(level_shared_buffer_size)

            tmp_segment_shared_buffer = np.array(segment_shared_buffer)
            tmp_segment_shared_buffer[1:, 1] += tmp_segment_shared_buffer[0:-1, 1]
            tmp_segment_total_shared_buffer = (tmp_segment_shared_buffer[:, 0]).sum() + (
            tmp_segment_shared_buffer[:, 1]).max()
            print("layer_shared_buffer_size: ", level_shared_buffer_size, tmp_segment_total_shared_buffer)

        print("operational intensity:  ", total_operations, total_dram_accesses, total_operations / total_dram_accesses)

        segment_latency = np.array(segment_latency).sum()
        segment_energy = np.array(segment_energy).sum()

        segment_shared_buffer = np.array(segment_shared_buffer)
        # print("segment_shared_buffersegment_shared_buffer  ", segment_shared_buffer.shape)
        # print(segment_shared_buffer, (segment_shared_buffer[:, 0]).sum())
        segment_shared_buffer[1:, 1] += segment_shared_buffer[0:-1, 1]
        segment_total_shared_buffer = (segment_shared_buffer[:, 0]).sum() + (segment_shared_buffer[:, 1]).max()
        print("segment_total_shared_buffer: ", segment_total_shared_buffer)
        if segment_total_shared_buffer > shared_buffer_size:
            print("OverSize global bufferrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
            print((segment_shared_buffer[:, 0]).sum(), (segment_shared_buffer[:, 1]).max())
        return segment_latency, segment_energy, segment_total_shared_buffer

def get_prime_factors(n):
    primes = defaultdict(int)
    primes['2'] = 0
    primes['3'] = 0
    primes['5'] = 0
    primes['7'] = 0
    while n % 2 == 0:
        primes['2'] += 1
        n = n // 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            primes[f'{i}'] += 1
            n = n // i
    if n > 2:
        primes[f'{n}'] += 1
    return primes

def config_arch_mapspace(in_config_dir='../../in_config', accelerator='simba', arch_file='simba_large',
                         map_file='mapspace', first_layer=True):
    with open(os.path.join(in_config_dir, accelerator, '{}.yaml'.format(arch_file)), 'r') as fd:
        arch = yaml.load(fd, Loader=yaml.SafeLoader)
    fd.close()

    shared_buffer_depth = arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth']
    shared_buffer_size = shared_buffer_depth * \
                         arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['block-size']

    run_config_dir = '../run_config'
    os.makedirs(os.path.join(run_config_dir, accelerator), exist_ok=True)
    with open(os.path.join(run_config_dir, accelerator, 'arch.yaml'), 'w') as fd:
        yaml.dump(arch, fd)
    fd.close()

    with open(os.path.join(in_config_dir, accelerator, '{}.yaml'.format(map_file)), 'r') as fd:
        mapspace = yaml.load(fd, Loader=yaml.SafeLoader)
    fd.close()
    if first_layer:
        mapspace['mapspace']['constraints'][-1]['bypass'] = []
        mapspace['mapspace']['constraints'][-1]['keep'] = ['Weights', 'Inputs', 'Outputs']
    else:
        mapspace['mapspace']['constraints'][-1]['bypass'] = ['Inputs']
        mapspace['mapspace']['constraints'][-1]['keep'] = ['Weights', 'Outputs']

    fd.close()

    with open(os.path.join(run_config_dir, accelerator, 'mapspace.yaml'), 'w') as fd:
        yaml.dump(mapspace, fd)
    fd.close()

    return shared_buffer_size



def tile_scheduling(segmentation_sol):
    return 'SRCKQPHN'


