#!/usr/bin/env python
# coding: utf-8

import gym
import argparse
import numpy as np
import datetime
from fourrooms import Fourrooms

from scipy.special import expit
# from scipy.misc import logsumexp
from scipy.special import logsumexp
from scipy.stats import entropy as cross_entropy
from sklearn.metrics import log_loss

import dill
import collections
import matplotlib.pyplot as plt
import threading
import os
import sys
from collections import defaultdict
import seaborn as sn
import pandas as pd
import pickle
import itertools



class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.weights.shape[1]))
        return int(np.argmax(self.value(phi)))

class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1., weight_init=None):
        self.rng = rng
        if weight_init is None:
            self.weights = np.zeros((nfeatures, nactions))
        else:
            self.weights = weight_init

        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))

class BoltzmannPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = 0.5*np.ones((nfeatures, nactions))
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        return int(self.rng.choice(self.nactions, p=self.pmf(phi)))


class SigmoidTermination:
    def __init__(self, rng, nfeatures, init=False):
        self.rng = rng
        if init:
            self.weights = np.random.rand(nfeatures,)
        else:
            self.weights = np.zeros((nfeatures,))

    def pmf(self, phi):
        return expit(np.sum(self.weights[phi]))

    def sample(self, phi):
        return int(self.rng.uniform() < self.pmf(phi))

    def grad(self, phi):
        terminate = self.pmf(phi)
        return terminate*(1. - terminate), phi

class IntraOptionQLearning:
    def __init__(self, discount, lr, terminations, weights):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights

    def start(self, phi, option):
        self.last_phi = phi
        self.last_option = option
        self.last_value = self.value(phi, option)

    def value(self, phi, option=None):
        if option is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, option], axis=0)

    def advantage(self, phi, option=None, entropy=1.0):
        values = self.value(phi)
        advantages = values - entropy*np.max(values)
        if option is None:
            return advantages
        return advantages[option]

    def update(self, phi, option, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Dense gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_option] += self.lr*tderror

        # Record new weights


        if not done:
            self.last_value = current_values[option]
        self.last_option = option
        self.last_phi = phi

        return tderror

class IntraOptionActionQLearning:
    def __init__(self, discount, lr, terminations, weights, qbigomega):
        self.lr = lr
        self.discount = discount
        self.terminations = terminations
        self.weights = weights
        self.qbigomega = qbigomega

    def value(self, phi, option, action):
        return np.sum(self.weights[phi, option, action], axis=0)

    def start(self, phi, option, action):
        self.last_phi = phi
        self.last_option = option
        self.last_action = action

    def update(self, phi, option, action, reward, done):
        # One-step update target
        update_target = reward
        if not done:
            current_values = self.qbigomega.value(phi)
            termination = self.terminations[self.last_option].pmf(phi)
            update_target += self.discount*((1. - termination)*current_values[self.last_option] + termination*np.max(current_values))

        # Update values upon arrival if desired
        tderror = update_target - self.value(self.last_phi, self.last_option, self.last_action)
        self.weights[self.last_phi, self.last_option, self.last_action] += self.lr*tderror

        self.last_phi = phi
        self.last_option = option
        self.last_action = action

        return tderror

class TerminationGradient:
    def __init__(self, terminations, critic, lr):
        self.terminations = terminations
        self.critic = critic
        self.lr = lr

    def update(self, phi, option, entropy, tdeoc, avg_entropy, std_entropy):
        magnitude, direction = self.terminations[option].grad(phi)
        entropy_norm = (entropy if std_entropy ==0 else (entropy - avg_entropy)/std_entropy)
        if tdeoc:
            self.terminations[option].weights[direction] -= self.lr*magnitude*(-entropy_norm) #*10 *(self.critic.value(phi, option)) # DEOC TERMINATION with value
        else:
            self.terminations[option].weights[direction] -= self.lr*magnitude*(self.critic.advantage(phi, option))

class IntraOptionGradient:
    def __init__(self, option_policies, lr):
        self.lr = lr
        self.option_policies = option_policies

    def update(self, phi, option, action, critic):
        actions_pmf = self.option_policies[option].pmf(phi)
        self.option_policies[option].weights[phi, :] -= self.lr*critic*actions_pmf
        self.option_policies[option].weights[phi, action] += self.lr*critic

class OneStepTermination:
    def sample(self, phi):
        return 1

    def pmf(self, phi):
        return 1.

class FixedActionPolicies:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

        return self.probs


# Generator to output pseudo reward
def get_op_entropy(phi, option_policies):
    cum_entropy = 0
    num_samples = min(len(option_policies), 6)
    combinations = list(itertools.combinations(range(len(option_policies)),2))
    for _ in range(num_samples):
        sample = combinations[rng.randint(0,len(combinations))]
        sampled_op1 = sample[0]
        sampled_op2 = sample[1]
        x1 = option_policies[sampled_op1].pmf(phi)
        x2 = option_policies[sampled_op2].pmf(phi)
        x1 = np.clip(x1,1e-20, 1.0)
        x2 = np.clip(x2,1e-20, 1.0)
        cum_entropy += -np.sum(x1*np.log(x2))/x1.shape[0]

    return cum_entropy/(num_samples)


# Create relevant folders and file name based on type of run
def get_path(args):
    file = "_nepisodes=" + str(args.nepisodes) \
           + "_nruns=" + str(args.nruns) \
           + "_nsteps=" + str(args.nsteps) \
           + "_noptions=" + str(args.noptions)

    file += "lr_intra={}_lr_term={}_lr_critic={}_lr_action_critic={}_temp={}".format(args.lr_intra, args.lr_term, args.lr_critic, args.lr_action_critic, args.temperature)

    if args.tdeoc:
        experiment_type = "TDEOC_"
    else:
        experiment_type = "Vanilla_OC_"

    if args.transfer:
        experiment_type += "transfer_"

    file = str(args.caption) + experiment_type + file
    # print(file)

    ## Save results
    results_dir = "Stats/" + str(file) + str(run_time_stamp) +  "/History"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plots_dir = "Stats/" + str(file) + str(run_time_stamp) + "/Plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    intra_weights_dir = "Stats/" + str(file) + str(run_time_stamp) + "/Intra_weights"
    if not os.path.exists(intra_weights_dir):
        os.makedirs(intra_weights_dir)

    term_weights_dir = "Stats/" + str(file) + str(run_time_stamp) + "/Term_weights"
    if not os.path.exists(term_weights_dir):
        os.makedirs(term_weights_dir)

    pol_weights_dir = "Stats/" + str(file) + str(run_time_stamp) + "/Pol_weights"
    if not os.path.exists(pol_weights_dir):
        os.makedirs(pol_weights_dir)

    episode_plots_dir = "Stats/" + str(file) + str(run_time_stamp) + "/Episode_plots"
    if not os.path.exists(episode_plots_dir):
        os.makedirs(episode_plots_dir)

    return file, results_dir, plots_dir, intra_weights_dir, term_weights_dir, pol_weights_dir, episode_plots_dir


# Returns the layout for fourrooms
def getEnvLayout():
    layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

    num_elem = 13
    line_count = 0
    mat_layout = -2*np.ones((num_elem, num_elem))
    mapping_state_row_col = defaultdict(list)
    ann_layout = np.zeros((num_elem, num_elem))

    state_count = 0
    for line in layout.splitlines():
        for i in range(num_elem):
            if line[i]!="w":
                mapping_state_row_col[state_count].extend([line_count, i])
                ann_layout[line_count, i] = state_count
                state_count +=1
            elif line[i]=="w":
                mat_layout[line_count, i] = 20
        line_count +=1

    return mat_layout, ann_layout, mapping_state_row_col


#Saving policy and intra_option policy distributions every few episodes
def render(run, episode, option_policies, policy, terminations, episode_plots_dir):

    pol_path = episode_plots_dir + "/Policy"
    intra_pol_path = episode_plots_dir + "/Intra_Policy"
    terminations_path = episode_plots_dir + "/Terminations"

    if(not os.path.isdir(pol_path)):
        os.makedirs(pol_path)
    if(not os.path.isdir(intra_pol_path)):
        os.makedirs(intra_pol_path)
    if(not os.path.isdir(terminations_path)):
        os.makedirs(terminations_path)

    # Plot Intra-Option Policies
    fig, ax = plt.subplots(len(option_policies), 4, figsize=(24, 20))
    for option in range(len(option_policies)):
        # print("************** Option: ", option)
        for action in range(4):
            mat_layout, ann_layout, mapping_state_row_col = getEnvLayout()
            wgt_pol = option_policies[option].weights[:, action]
            # print ("Action: ",action)
            wgt_pol = (wgt_pol-min(wgt_pol))/(max(wgt_pol)-min(wgt_pol))
            #         fig, ax = plt.subplots(figsize=(6, 5))
            for curr_state in range(env.observation_space.n):
                r, c = mapping_state_row_col[curr_state]
                mat_layout[r,c] = wgt_pol[curr_state]*15

            sn.set(font_scale=0.6)
            sn.heatmap(mat_layout, ax= ax[option, action], fmt = '', cmap="YlGnBu", cbar = False, square=True)
            ax[option, action].set_title(label ="Option {} Action {}".format(option+1, action+1), fontdict= {'fontsize': 25})

    plt.savefig(os.path.join(intra_pol_path, "Run {} Episode-{}".format(run+1, episode+1)))
    # plt.show()
    plt.close()

    # Plot Policy over options
    fig, ax = plt.subplots(1, len(option_policies), figsize=(24, 20))
    for option in range(len(option_policies)):
        mat_layout, ann_layout, mapping_state_row_col = getEnvLayout()
        wgt_pol = policy.weights[:, option]
        # print ("Option: ",option)
        wgt_pol = (wgt_pol-min(wgt_pol))/(max(wgt_pol)-min(wgt_pol))
        #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(6, 5))
        for curr_state in range(env.observation_space.n):
            r, c = mapping_state_row_col[curr_state]
            mat_layout[r,c] = wgt_pol[curr_state]*15

        sn.set(font_scale=0.6)
        sn.heatmap(mat_layout, ax=ax[option], fmt = '', cmap="YlGnBu", cbar = False, square=True)
        ax[option].set_title(label ="Option {}".format(option+1), fontdict= {'fontsize': 30})

    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    plt.savefig(os.path.join(pol_path, "Run {} Episode-{}".format(run+1, episode+1)))
    # plt.show()
    plt.close()

    # Plot terminations
    fig, ax = plt.subplots(1, len(terminations), figsize=(24, 20))
    for option in range(len(terminations)):
        mat_layout, ann_layout, mapping_state_row_col = getEnvLayout()
        wgt_term = terminations[option].weights[:]
        # print ("Option: ",option)
        wgt_term = (wgt_term-min(wgt_term))/(max(wgt_term)-min(wgt_term))
        for curr_state in range(env.observation_space.n):
            r, c = mapping_state_row_col[curr_state]
            mat_layout[r,c] = wgt_term[curr_state]*15

        sn.set(font_scale=0.6)
        sn.heatmap(mat_layout, ax=ax[option], fmt = '', cmap="YlGnBu", cbar = False, square=True)
        ax[option].set_title(label ="Option {}".format(option+1), fontdict= {'fontsize': 30})


    plt.savefig(os.path.join(terminations_path, "Run {} Episode-{}".format(run+1, episode+1)))
    # plt.show()
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--discount', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_term', help="Termination gradient learning rate", type=float, default=5e-2)
    parser.add_argument('--lr_intra', help="Intra-option gradient learning rate", type=float, default=1e-2)
    parser.add_argument('--lr_critic', help="Learning rate", type=float, default=5e-1)
    parser.add_argument('--lr_action_critic', help="Learning rate", type=float, default=5e-1)
    parser.add_argument('--epsilon', help="Epsilon-greedy for policy over options", type=float, default=1e-3)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=300)
    parser.add_argument('--nsteps', help="Maximum number of steps per episode", type=int, default=1000)
    parser.add_argument('--noptions', help='Number of options', type=int, default=4)
    parser.add_argument('--baseline', help="Use the baseline for the intra-option gradient", default=False, action='store_true')
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1e-3)
    parser.add_argument('--primitive', help="Augment with primitive", default=False, action='store_true')
    parser.add_argument('--transfer', help="Transfer goal to a random one", default=True, action='store_true')
    parser.add_argument('--tdeoc', help="Diversity driven terminations", default=False, action='store_true')


    parser.add_argument('--render', help="Render actions", default=False, action='store_true')
    parser.add_argument('--caption', help='Adding a prefix to run name', type=str, default='')

    args = parser.parse_args()

    ### When args give incomplete information
    rng = np.random.RandomState(1234)
    env = gym.make('Fourrooms-v0')

    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n


    run_time_stamp = datetime.datetime.now()
    history = np.zeros((args.nruns, args.nepisodes, 3))
    intra_pol_weights = np.zeros((args.nruns, args.noptions, nfeatures, nactions))
    pol_weights = np.zeros((args.nruns, nfeatures, args.noptions))
    termination_weights = np.zeros((args.nruns, args.noptions, nfeatures))

    #Save Stats/Dir
    file, results_dir, plots_dir, intra_weights_dir, term_weights_dir, pol_weights_dir, episode_plots_dir = get_path(args)
    print(file)

    tradeoff= 0.0001
    deoc= False



    for run in range(args.nruns):

        # Room states for possible goals
        room1_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
        room2_goals = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 36, 37, 38, 39, 40, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56]
        room3_goals = [57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 83, 84, 85, 86, 87, 94, 95, 96, 97, 98]
        room4_goals = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 41, 42,43, 44, 45]
        rooms = np.array([room1_goals, room2_goals, room3_goals, room4_goals])


        # The intra-option policies are linear-softmax functions
        option_policies = [SoftmaxPolicy(rng, nfeatures, nactions, args.temperature) for op in range(args.noptions)]
        if args.primitive:
            option_policies.extend([FixedActionPolicies(act, nactions) for act in range(nactions)])

        # The termination function are linear-sigmoid functions
        option_terminations = [SigmoidTermination(rng, nfeatures) for _ in range(args.noptions)]
        if args.primitive:
            option_terminations.extend([OneStepTermination() for _ in range(nactions)])

        # E-greedy policy over options
        #policy = EgreedyPolicy(rng, nfeatures, args.noptions, args.epsilon)
        policy = SoftmaxPolicy(rng, nfeatures, args.noptions, args.temperature)

        # Different choices are possible for the critic. Here we learn an
        # option-value function and use the estimator for the values upon arrival
        critic = IntraOptionQLearning(args.discount, args.lr_critic, option_terminations, policy.weights)

        # Learn Qomega separately
        action_weights = np.zeros((nfeatures, args.noptions, nactions))
        action_critic = IntraOptionActionQLearning(args.discount, args.lr_action_critic, option_terminations, action_weights, critic)

        # Improvement of the termination functions based on gradients
        termination_improvement= TerminationGradient(option_terminations, critic, args.lr_term)

        # Intra-option gradient improvement with critic estimator
        intraoption_improvement = IntraOptionGradient(option_policies, args.lr_intra)

        avg_entropy, std_entropy = 0.0, 1.0
        sum, samples = 0.0, 0
        entropys = np.zeros(80)

        for episode in range(args.nepisodes):

            if args.transfer:
                if episode == 1000:
                    # room = np.random.randint(0,4)
                    env.goal = np.random.choice(rooms[0])
                    print("Goal shifted to {}".format(env.goal))

            phi = features(env.reset())
            option = policy.sample(phi)
            action = option_policies[option].sample(phi)
            critic.start(phi, option)
            action_critic.start(phi, option, action)

            cumreward = 0.
            duration = 1
            option_switches = 0
            avgduration = 0.
            last_step = 0
            cum_pseudo_reward = 0
            cum_entropy = 0

            for step in range(args.nsteps):
                observation, reward, done, _ = env.step(action)
                phi = features(observation)

                # Termination might occur upon entering the new state
                if option_terminations[option].sample(phi):
                    # if True:
                    option = policy.sample(phi)
                    option_switches += 1
                    avgduration += (1./option_switches)*(duration - avgduration)
                    duration = 1

                action = option_policies[option].sample(phi)


                entropy_loss = get_op_entropy(phi, option_policies)
                pseudo_reward = (1-tradeoff)*reward + tradeoff*entropy_loss
                cum_pseudo_reward += pseudo_reward
                cum_entropy += entropy_loss

                entropys[samples%len(entropys)] = entropy_loss

                sum += entropy_loss
                samples += 1

                if deoc:
                    # Critic update
                    td_critic = critic.update(phi, option, pseudo_reward, done)

                    td_action_critic = action_critic.update(phi, option, action, pseudo_reward, done)
                else:
                    # Critic update
                    td_critic = critic.update(phi, option, reward, done)

                    td_action_critic = action_critic.update(phi, option, action, reward, done)



                if isinstance(option_policies[option], SoftmaxPolicy):
                    # Intra-option policy update
                    critic_feedback = action_critic.value(phi, option, action)
                    if args.baseline:
                        critic_feedback -= critic.value(phi, option)
                    intraoption_improvement.update(phi, option, action, critic_feedback)

                    avg_entropy = np.mean(entropys)
                    std_entropy = np.std(entropys)
                    # Termination update
                    termination_improvement.update(phi, option, entropy_loss, args.tdeoc, sum/samples, 1.0)

                cumreward += reward
                duration += 1
                last_step = step
                if done:
                    break
            # avg_entropy = np.mean(np.array(entropys))
            # std_entropy = np.std(np.array(entropys))

            if episode%100 == 0:
                out = 'Run {} episode {} steps {} cumreward {} '.format(run, episode, (step), cumreward)
                out += '|| avg_entropy {} std_entropy {} || '.format(sum/samples, std_entropy)
                out += 'avg. duration {} switches {}'.format(avgduration, option_switches)
                print(out)


            if args.render and (episode == args.nepisodes-1 or episode == 999):
                render(run, episode, option_policies, policy, option_terminations, episode_plots_dir)


            history[run, episode, 0] = step
            history[run, episode, 1] = cum_pseudo_reward/(last_step+1)
            history[run, episode, 2] = cum_entropy/(last_step+1)

        # [intra_pol_weights[op_pol].append(option_policies[op_pol].weights) for op_pol in range(len(option_policies))]
        intra_pol_weights[run] = np.array(([op.weights for op in option_policies]))
        pol_weights[run] = policy.weights
        termination_weights[run] = np.array(([op.weights for op in option_terminations]))
        print()

        sum, samples = 0.0,0.0
        entropys = np.zeros(80)
    print(file)
    print("Run Time: {}".format(datetime.datetime.now() - run_time_stamp))
    ### Save stats for every run

    np.save(os.path.join(results_dir, "history"), np.array(history))
    np.save(os.path.join(intra_weights_dir, "intra_pol_weights"), np.array(intra_pol_weights))
    np.save(os.path.join(pol_weights_dir, "pol_weights"), np.array(pol_weights))
    np.save(os.path.join(term_weights_dir, "term_weights"), np.array(termination_weights))


    # Plot results
    for i in range(1):
        mean = np.mean(history[:,:,0], axis =0)
        std = np.std(history[:,:,0], axis =0)
        x = np.arange(len(mean))
        plt.ylim(0,args.nsteps/2)
        plt.xlim(0, args.nepisodes)
        plt.plot(x, mean, label="")

    plt.title("Tuning {}".format(args.caption))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(os.path.join(plots_dir, "plot") + ".png")
    plt.close()