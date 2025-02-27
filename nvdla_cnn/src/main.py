import os
import yaml
import pickle
import re

import math
import numpy as np
import random
import copy
from datetime import datetime

from segmentation import Workload


def inter_op_dse():

    layer2budgets = {112: None, 56: None, 28: None, 14: None, 7: None}

    layer2budgets[112] = [np.array([2, 0, 0, 1], dtype=np.int32),
                          np.array([2, 0, 0, 1], dtype=np.int32),
                          np.array([2, 0, 0, 0], dtype=np.int32),
                          np.array([6, 0, 0, 0], dtype=np.int32),
                          np.array([6, 0, 0, 0], dtype=np.int32),
                          np.array([0, 0, 0, 0], dtype=np.int32)]

    layer2budgets[56] = [np.array([0, 0, 0, 1], dtype=np.int32),
                         np.array([0, 0, 0, 1], dtype=np.int32),
                         np.array([8, 0, 0, 0], dtype=np.int32),
                         np.array([8, 0, 0, 0], dtype=np.int32),
                         np.array([8, 0, 0, 0], dtype=np.int32),
                         np.array([0, 0, 0, 0], dtype=np.int32)]

    layer2budgets[28] = [np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([4, 0, 0, 0], dtype=np.int32),
                         np.array([2, 0, 0, 0], dtype=np.int32),
                         np.array([4, 0, 0, 0], dtype=np.int32),
                         np.array([2, 0, 0, 0], dtype=np.int32)]

    layer2budgets[14] = [np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([4, 0, 0, 0], dtype=np.int32),
                         np.array([2, 0, 0, 0], dtype=np.int32),
                         np.array([4, 0, 0, 0], dtype=np.int32),
                         np.array([2, 0, 0, 0], dtype=np.int32)]

    layer2budgets[7] = [np.array([0, 0, 0, 1], dtype=np.int32),
                        np.array([0, 0, 0, 1], dtype=np.int32),
                        np.array([5, 0, 0, 0], dtype=np.int32),
                        np.array([2, 0, 0, 0], dtype=np.int32),
                        np.array([4, 0, 0, 0], dtype=np.int32),
                        np.array([3, 0, 0, 0], dtype=np.int32)]

    layer2budgets[112] = [np.array([2, 0, 0, 1], dtype=np.int32),
                          np.array([2, 0, 0, 1], dtype=np.int32),
                          np.array([12, 0, 0, 0], dtype=np.int32),
                          np.array([16, 0, 0, 0], dtype=np.int32),
                          np.array([16, 0, 0, 0], dtype=np.int32),
                          np.array([0, 0, 0, 0], dtype=np.int32)]

    layer2budgets[56] = [np.array([0, 0, 0, 1], dtype=np.int32),
                         np.array([0, 0, 0, 1], dtype=np.int32),
                         np.array([18, 0, 0, 0], dtype=np.int32),
                         np.array([18, 0, 0, 0], dtype=np.int32),
                         np.array([18, 0, 0, 0], dtype=np.int32),
                         np.array([0, 0, 0, 0], dtype=np.int32)]

    layer2budgets[28] = [np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([14, 0, 0, 0], dtype=np.int32),
                         np.array([12, 0, 0, 0], dtype=np.int32),
                         np.array([14, 0, 0, 0], dtype=np.int32),
                         np.array([2, 0, 0, 0], dtype=np.int32)]

    layer2budgets[14] = [np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([1, 0, 0, 1], dtype=np.int32),
                         np.array([14, 0, 0, 0], dtype=np.int32),
                         np.array([12, 0, 0, 0], dtype=np.int32),
                         np.array([14, 0, 0, 0], dtype=np.int32),
                         np.array([2, 0, 0, 0], dtype=np.int32)]

    layer2budgets[7] = [np.array([0, 0, 0, 1], dtype=np.int32),
                        np.array([0, 0, 0, 1], dtype=np.int32),
                        np.array([15, 0, 0, 0], dtype=np.int32),
                        np.array([12, 0, 0, 0], dtype=np.int32),
                        np.array([14, 0, 0, 0], dtype=np.int32),
                        np.array([3, 0, 0, 0], dtype=np.int32)]

    # layer2budgets_list = []
    # layer2budgets_list.append(copy.deepcopy(layer2budgets))
    #
    # layer2budgets[112] = [np.array([1, 0, 0, 1], dtype=np.int32),
    #                       np.array([1, 0, 0, 1], dtype=np.int32),
    #                       np.array([0, 2, 1, 0], dtype=np.int32),
    #                       np.array([6, 2, 1, 1], dtype=np.int32),
    #                       np.array([6, 2, 1, 1], dtype=np.int32),
    #                       np.array([0, 0, 0, 0], dtype=np.int32)]
    #
    # layer2budgets[56] = [np.array([0, 0, 0, 1], dtype=np.int32),
    #                      np.array([0, 0, 0, 1], dtype=np.int32),
    #                      np.array([8, 2, 1, 0], dtype=np.int32),
    #                      np.array([8, 2, 1, 1], dtype=np.int32),
    #                      np.array([8, 2, 1, 1], dtype=np.int32),
    #                      np.array([0, 0, 0, 0], dtype=np.int32)]
    #
    # layer2budgets[28] = [np.array([1, 0, 0, 1], dtype=np.int32),
    #                      np.array([1, 0, 0, 1], dtype=np.int32),
    #                      np.array([4, 2, 1, 0], dtype=np.int32),
    #                      np.array([2, 2, 1, 1], dtype=np.int32),
    #                      np.array([4, 2, 1, 1], dtype=np.int32),
    #                      np.array([2, 0, 0, 0], dtype=np.int32)]
    #
    # layer2budgets[14] = [np.array([1, 0, 0, 1], dtype=np.int32),
    #                      np.array([1, 0, 0, 1], dtype=np.int32),
    #                      np.array([4, 2, 1, 0], dtype=np.int32),
    #                      np.array([2, 2, 1, 1], dtype=np.int32),
    #                      np.array([4, 2, 1, 1], dtype=np.int32),
    #                      np.array([2, 0, 0, 0], dtype=np.int32)]
    #
    # layer2budgets[7] = [np.array([0, 0, 0, 1], dtype=np.int32),
    #                     np.array([0, 0, 0, 1], dtype=np.int32),
    #                     np.array([5, 2, 1, 0], dtype=np.int32),
    #                     np.array([2, 2, 1, 1], dtype=np.int32),
    #                     np.array([4, 2, 1, 1], dtype=np.int32),
    #                     np.array([3, 0, 0, 0], dtype=np.int32)]
    return layer2budgets


if __name__ == '__main__':

    # for problems in ['resnet50', 'resnet18']:
    #     for batch_size in [1, 8]:
    #         RL4Map(problems, batch_size)
    # GA4InterMap('dwd')
    accelerator = 'nvdla'
    architectures = ['nvdla_dramband2_1MB_SMEMband']
    # batch_size = 16
    # architecture = 'simba_cloud'
    for architecture in architectures:
        # for problems in ['resnet50', 'mobilenetv2']:
        for workload_name in ['resnet50', 'mobilenetv2', 'efficientnet']:
        # for workload_name in ['efficientnet']:
        # for problems in ['mobilenetv2']:
        # for problems in ['gpt3_8k']:
            for batch_size in [16, 1]:
                with open(os.path.join('../../in_config', accelerator, '{}.yaml'.format(architecture)), 'r') as fd:
                    arch = yaml.load(fd, Loader=yaml.SafeLoader)
                fd.close()

                shared_buffer_depth = arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth']
                shared_buffer_size = shared_buffer_depth * \
                                     arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes'][
                                         'block-size']

                best_latency = float('inf')
                best_energy = float('inf')
                best_edp = float('inf')
                best_segmentation_sol = None
                best_P_Q_C_K_H_N_sol = None

                layer2budgets = inter_op_dse()

                workload_dir = '../../benchmarks/'
                workload = Workload(workload_dir, workload_name, batch_size, layer2budgets)

                segmentation_sol = workload.greedy_segmentation(shared_buffer_size)

                save_dir = os.path.join('../report_dir', '{}_order_KHPQN_fuse'.format(architecture),
                                        '{}'.format(workload_name),
                                        'batchsize{}'.format(batch_size))
                os.makedirs(save_dir, exist_ok=True)

                if segmentation_sol is not None:
                    print(len(segmentation_sol))

                    total_latency = 0
                    total_energy = 0

                    segmentation_str = '_'.join(str(x) for x in segmentation_sol)
                    report_dir = os.path.join(save_dir, 'segmentation_sol{}'.format(segmentation_str))
                    print(report_dir)
                    os.makedirs(report_dir, exist_ok=True)
                    fw = open(os.path.join(report_dir, 'latency-energy.txt'), 'w')

                    for seg_idx, segment_start in enumerate(segmentation_sol[:-1]):
                        # if sub_idx == len(subgraph_partition_sol) - 1:
                        #     subgraph_end = len(layers)
                        # else:
                        print(segment_start)
                        segment_end = segmentation_sol[seg_idx + 1]\

                        if segment_start == segment_end - 1:
                            pre_tile_budgets = None
                        else:
                            pre_tile_budgets = layer2budgets

                        segment_latency, segment_energy, segment_shared_buffer \
                            = workload.segment_perf(accelerator, architecture, report_dir, segment_start, segment_end)

                        total_latency += segment_latency
                        total_energy += segment_energy

                        print("segment_latency: ", segment_latency,
                              "segment_energy: ", segment_energy,
                              "total_latency: ", total_latency,
                              "total_energy: ", total_energy)
                        print("segment_shared_buffer: ", np.array(segment_shared_buffer).sum(), np.array(segment_shared_buffer))
                        # fw.write(str(segment_off_latency) + '\n')
                        # fw.write(str(segment_on_latency) + '\n')
                        fw.write(str(segment_latency) + '\n')
                        fw.write(str(total_latency) + '\n')

                    fw.write(str(total_energy) + '\n')
                    fw.write(str(total_latency * total_energy) + '\n')
                    fw.close()

                    if total_latency < best_latency:
                        best_latency = total_latency
                        best_energy = total_energy
                        best_edp = total_latency * total_energy
                        best_segmentation_sol = segmentation_sol
                        # best_P_Q_C_K_H_N_sol = np.stack((P_budgets, Q_budgets, C_budgets, K_budgets, H_budgets, N_budgets), axis=0)

                fw = open(os.path.join(save_dir, 'latency-energy-sol.txt'), 'w')
                fw.write(str(best_latency) + '\n')
                fw.write(str(best_energy) + '\n')
                fw.write(str(best_latency * best_energy) + '\n')
                fw.write(str(best_segmentation_sol) + '\n')
                # fw.write(str(best_P_Q_C_K_H_N_sol[0]) + '\n')
                # fw.write(str(best_P_Q_C_K_H_N_sol[1]) + '\n')
                # fw.write(str(best_P_Q_C_K_H_N_sol[2]) + '\n')
                # fw.write(str(best_P_Q_C_K_H_N_sol[3]) + '\n')
                # fw.write(str(best_P_Q_C_K_H_N_sol[4]) + '\n')
                fw.close()


# TODO
# different budgets for inter-segment
# different budgets for intra-segment
# graph, bfs

