import torch
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
import argparse

from env import Environment
from actor import Actor
from transformer.Optim import ScheduledOptim


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def compute_policy_loss(rewards, log_probs, log_prob_masks):
    '''
    :param rewards: length,batch
    :param log_probs: length,batch,5
    :param log_prob_masks: length,batch,5
    :return:
    '''
    dis_rewards = []
    gamma = 0.99
    num_episodes = log_probs.size(1)
    batch_masks = log_probs.new_ones(num_episodes)
    success_idx = []
    fail_idx = []
    for i in range(num_episodes):
        if rewards[-1, i] > 0:
            success_idx.append(i)
        else:
            fail_idx.append(i)
    if len(fail_idx) > 3*len(success_idx):
        fail_idx = random.sample(fail_idx, 3*len(success_idx))
    # print(len(success_idx), len(fail_idx), rewards[-1, :])
    # batch_masks = log_probs.new_zeros(num_episodes)
    # batch_masks[success_idx] = 1.
    # batch_masks[fail_idx] = 1.

    rewards = rewards[7:]
    log_probs = log_probs[:-7]
    log_prob_masks = log_prob_masks[:-7]

    R = np.zeros(num_episodes)
    for r in rewards[::-1]:
        R = r + gamma * R
        dis_rewards.insert(0, R)
    dis_rewards = torch.from_numpy(np.array(dis_rewards)).to(log_probs.device)
    # print(dis_rewards.size(), log_prob_masks[:,7,:], log_prob_masks[:,6,:])
    policy_loss = dis_rewards * (-1 * log_probs * log_prob_masks).sum(dim=-1)
    policy_loss = policy_loss.sum(dim=0) * batch_masks
    policy_loss = policy_loss.sum() / batch_masks.sum()

    return policy_loss


def intra_op_mapping_dse(accelerator, problem, report_dir, gb_tile_budgets):
    parser = argparse.ArgumentParser()
    parser.add_argument('--intra_map_epochs', type=int, default=5)
    parser.add_argument('--density', type=str, default='0.5,1,1', help='The density of Input, Output, Weight Tenor')
    parser.add_argument('--save_chkpt', action='store_true', default=True, help='Create a checkpoint when finished')
    parser.add_argument('--use_sparse', action='store_true', default=False,
                        help='Execute Map Space Exploration on sparse accelerator')
    parser.add_argument('--explore_bypass', action='store_true', default=False,
                        help='Enable it can add bypass buffer option in to the search space')

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner', type=int, default=1024)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)

    parser.add_argument('--n_warmup_steps', type=int, default=2)
    parser.add_argument('--lr_mul', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)

    # print(report_dir)
    fitness = ['latency', 'latency', 'energy']
    print(f'Fitness Objective: {fitness}')
    opt = parser.parse_args()

    set_seed(opt.seed)
    density = opt.density.split(',')
    density = {'Inputs': float(density[0]), 'Outputs': float(density[1]), 'Weights': float(density[2])}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = Environment(gb_tile_budgets,
                      accelerator=accelerator, fitness_obj=fitness, report_dir=report_dir, use_pool=True, use_IO=True,
                      save_chkpt=opt.save_chkpt, num_episodes=32)
    tile_budgets = torch.from_numpy(env.tile_remain_budgets).to(device)
    actor = Actor(opt.d_model, opt.d_inner, opt.n_layers, opt.n_head, opt.d_k, opt.d_v, accelerator,
                  env.buf_spmap_cstr, env.buffer_size_list, env.steps_per_level,
                  problem['problem']['instance'], env.prime2idx, tile_budgets, gb_tile_budgets).to(device)
    # if actor_state_dict is not None:
    #     actor.load_state_dict(actor_state_dict)
    actor.train()

    optimizer = ScheduledOptim(
        optim.Adam(actor.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    # optimizer = optim.Adam(actor.parameters(), lr=1e-3, betas=(0.9, 0.999))

    for ep in range(opt.intra_map_epochs):
        # print('Epoch {}'.format(ep))
        # print('epoch start : ', datetime.now())
        state_info = env.reset()
        actor.reset()
        total_log_probs = []
        total_log_prob_masks = []
        total_rewards = []
        optimizer.zero_grad()
        for step in range(env.total_steps):
            program_seq = torch.from_numpy(state_info[0]).type(torch.LongTensor).to(device)
            order_mask = torch.from_numpy(state_info[1]).type(torch.FloatTensor).to(device)
            tile_remain_budgets = torch.from_numpy(state_info[2]).type(torch.LongTensor).to(device)
            tile_masks = torch.from_numpy(state_info[3]).type(torch.FloatTensor).to(device)
            mode = state_info[4]
            cur_buffer_level = state_info[5]
            program_seq_disorder = torch.from_numpy(state_info[6]).type(torch.LongTensor).to(device)
            step_actions, step_log_probs, step_log_prob_masks = actor(program_seq, order_mask, tile_remain_budgets,
                                                                      tile_masks, mode, cur_buffer_level,
                                                                      program_seq_disorder)
            state_info, reward, done, info = env.step(step_actions)
            total_rewards.append(reward)
            total_log_probs.append(step_log_probs)
            total_log_prob_masks.append(step_log_prob_masks)

            if done:
                break

        total_rewards = np.stack(total_rewards, axis=0)
        total_log_probs = torch.stack(total_log_probs, dim=0)
        total_log_prob_masks = torch.stack(total_log_prob_masks, dim=0)
        policy_loss = compute_policy_loss(total_rewards, total_log_probs, total_log_prob_masks)
        policy_loss.backward()
        # for param_group in optimizer._optimizer.param_groups:
        #     # param_group['lr'] *= 0.8
        #     print(param_group['lr'])
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), 10)
        # optimizer.step()
        optimizer.step_and_update_lr()

        chkpt = env.record_chkpt(ep == opt.intra_map_epochs - 1)

    env.clean_timeloop_output_files()
    return chkpt