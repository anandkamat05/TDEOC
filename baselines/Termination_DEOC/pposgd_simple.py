from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import os
from scipy.special import softmax

import cv2
import baselines.Termination_DEOC.cnn_policy as cnn
import itertools
import sys

def traj_segment_generator(pi, env, horizon, stochastic, num_options,saves,results,rewbuffer,dc):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ent_ret = 0 ## pseudo rewards in current episodes
    cur_op_ent_ret = 0 ## Intra option entropy in current episodes
    cur_ep_len = 0 # len of current episode
    cur_opt_switches = 0
    ep_rets = [] # returns of completed episodes in this segment
    ent_rets = []
    op_ent_rets=[]
    ep_lens = [] # lengths of ...
    opt_switches = []
    iters_so_far=0

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    prews = np.zeros((horizon), 'float32')
    vpreds = np.zeros(horizon, 'float32')
    vpreds_ent = np.zeros(horizon, 'float32') #Entropy Value
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()


    option = pi.get_option(ob)

    optpol_p=[]    
    term_p=[]
    value_val=[]
    opt_duration = [[] for _ in range(num_options)]
    logstds = [[] for _ in range(num_options)]
    curr_opt_duration = 0.

    while True:
        prevac = ac
        if isinstance(pi, cnn.CnnPolicy):
            ac, vpred, vpred_ent, feats = pi.act(stochastic, ob, option)
            logstd=0
        else:
            ac, vpred, vpred_ent, feats,logstd = pi.act(stochastic, ob, option)
        logstds[option].append(logstd)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "realrew": realrews, "prew": prews, "vpred" : vpreds, "vpred_ent" : vpreds_ent, "new" : news,
                    "ac" : acs, "opts" : opts, "prevac" : prevacs, "nextvpred": vpred * (1 - new), "nextvpred_ent": vpred_ent * (1 - new),
                    "ep_rets" : ep_rets, "ent_rets": ent_rets, "ep_lens" : ep_lens, 'term_p': term_p, 'value_val': value_val,
                     "opt_dur": opt_duration, "optpol_p":optpol_p, "logstds": logstds, "opt_switches":opt_switches}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ent_rets = [] #### Storing pseudo rewards for episodes
            ep_lens = []
            term_p  = []
            value_val=[]
            opt_duration = [[] for _ in range(num_options)]
            logstds = [[] for _ in range(num_options)]
            curr_opt_duration = 0.
            iters_so_far +=1


            if iters_so_far == 100:
                if hasattr(env,'NAME'):
                    print("#################### Switching Goal ########################")
                    env.change()

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        vpreds_ent[i] = vpred_ent
        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac


        ob, rew, new, _ = env.step(ac)
        if not hasattr(env,'NAME'):
            if env.spec.id[:-3].lower()[:9] == "miniworld":
                env.set_usertxt(option)
        # env.render()

        prew = gen_pseudo_reward(pi, ob, num_options, stochastic)
        if hasattr(env,'NAME'):
            prew = prew*1e-2
        elif env.spec.id[:-3].lower()[:9] == "miniworld":
            prew = prew*1e-2
        else:
            prew = prew*1e-1  # Changing from 1e-2 while switching to softmax

        # rew = rew/10 if (env.spec.id[:-3].lower() == 'humanoid' or env.spec.id[:-3].lower() == 'humanoidstandup') else rew  ## Scaling reward down further for humanoid
        rew = rew/10 if num_options > 1 and not hasattr(env,'NAME') and not env.spec.id[:-3].lower()[:9] == "miniworld" else rew # To stabilize learning.

        prews[i] = prew
        rews[i] = rew
        realrews[i] = rew

        curr_opt_duration += 1

        ### Book-keeping
        t_p = []
        v_val = []
        for oopt in range(num_options):
            v_val.append(pi.get_vpred([ob],[oopt])[0][0])
            t_p.append(pi.get_tpred([ob],[oopt])[0][0])
        term_p.append(t_p)
        optpol_p.append(pi._get_op([ob])[0][0])
        value_val.append(v_val)
        term = pi.get_term([ob],[option])[0][0]
        ###

        if term:
            if num_options > 1:
                rews[i] -= dc            
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            old_option = option
            option = pi.get_option(ob)
            if option != old_option:
                cur_opt_switches += 1

        # rew = rew*10 if (env.spec.id[:-3].lower() == 'humanoid' or env.spec.id[:-3].lower() == 'humanoidstandup') else rew
        cur_ep_ret += rew*10 if num_options > 1 and not hasattr(env,'NAME') and not env.spec.id[:-3].lower()[:9] == "miniworld" else rew
        cur_ent_ret += prew
        cur_ep_len += 1


        if new:
            ep_rets.append(cur_ep_ret)
            ent_rets.append(cur_ent_ret)
            ep_lens.append(cur_ep_len)
            opt_switches.append(cur_opt_switches)

            cur_opt_switches =0
            cur_ep_ret = 0
            cur_ent_ret =0
            cur_ep_len = 0
            ob = env.reset()
            option = pi.get_option(ob)
        t += 1


def gen_pseudo_reward(pi, ob, num_options, stochastic=True):
    if num_options==1:
        return 0
    else:
        cum_entropy = 0
        combinations = list(itertools.combinations(range(num_options), 2))
        for i in range(len(combinations)):
            sampled_op1 = combinations[i][0]
            sampled_op2 = combinations[i][1]
            if isinstance(pi, cnn.CnnPolicy):
                # x1, _, _, _= pi.act(stochastic, ob, sampled_op1)
                # x2, _, _, _ = pi.act(stochastic, ob, sampled_op2)
                logits_op1 = pi.get_logits(stochastic, ob, sampled_op1)[0]
                logits_op2 = pi.get_logits(stochastic, ob, sampled_op2)[0]
                pd_op1 = softmax(logits_op1)
                pd_op2 = softmax(logits_op2)
                cum_entropy += -np.sum(pd_op1*np.log(pd_op2))/(10*pd_op1.shape[0])

            else:
                x1, _, _, _,_ = pi.act(stochastic, ob, sampled_op1)
                x2, _, _, _, _ = pi.act(stochastic, ob, sampled_op2)
                x1 = softmax(x1)
                x2 = softmax(x2)
                x1 = np.clip(x1,1e-20, 1.0)
                x2 = np.clip(x2,1e-20, 1.0)
                cum_entropy += -np.sum(x1*np.log(x2))/x1.shape[0]

        cum_entropy = cum_entropy/len(combinations)
        return cum_entropy*2


def add_vtarg_and_adv(seg, gamma, lam, deoc=False, tradeoff=0.1):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    T = len(seg["rew"])
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    vpred_ent = (np.append(seg["vpred_ent"], seg["nextvpred_ent"]) if deoc else np.zeros((T+T), 'float32'))
    seg["adv"] = gaelam = np.empty(T, 'float32')
    seg["adv_ent"] = gaelam_ent = np.zeros(T, 'float32')
    rew = seg["rew"]
    prew = seg["prew"]
    lastgaelam = 0
    lastgaelam_ent =0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]

        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]  # Vanilla TD Error

        # Changes for deoc
        if deoc:
            delta +=  (1-tradeoff)*rew[t] + (tradeoff)*prew[t] - rew[t]   # Compute changed TD_error
            delta_ent = (1-tradeoff)*rew[t] + gamma * vpred_ent[t+1] * nonterminal - vpred_ent[t]
            gaelam_ent[t] = lastgaelam_ent = delta_ent + gamma * lam * nonterminal * lastgaelam_ent + (tradeoff)*prew[t]


        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    if deoc:
        seg["tdlamret_ent"] = seg["adv_ent"] + seg["vpred_ent"]
    else:
        seg["tdlamret_ent"] =  np.zeros(T, 'float32')

    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        num_options=1,
        app='',
        saves=False,
        wsaves=False,
        epoch=-1,
        seed=1,
        dc=0,
        render=False,
        caption='',
        deoc=False,
        tradeoff=0.1,
        term_mult=1.0,
        lr_mult=1.0,
        tdeoc = False
        ):


    optim_batchsize_ideal = optim_batchsize 
    np.random.seed(seed)
    tf.random.set_seed()(seed)
    env.seed(seed)

    ### Book-keeping
    if hasattr(env,'NAME'):
        gamename = env.NAME.lower()
    else:
        gamename = env.spec.id[:-3].lower()

    gamename += '-seed' + str(seed)
    gamename += app 

    dirname = '{}_{}opts_saves/'.format(gamename,num_options)

    ### More book-kepping
    results=[]
    if tdeoc:
        results_name = caption + 'TDEOC_' + gamename + '_tradeoff'+ str(tradeoff) + '_dc' + str(dc)
    elif deoc:
        results_name = caption + 'DEOC_' + gamename + '_tradeoff'+ str(tradeoff) + '_dc' + str(dc)
    else:
        results_name = caption + 'Vanilla_' +  gamename  + '_dc' + str(dc)

    if epoch >= 0:
        results_name_file = results_name + '_epoch' + str(epoch)
        results_name_file += '_term_mult' + str(term_mult) + '_lr_mult' + str(lr_mult) +  '_'+str(num_options)+'opts' + '_results.csv'

    results_name += '_term_mult' + str(term_mult) + '_lr_mult' + str(lr_mult) +  '_'+str(num_options)+'opts' + '_results.csv'

    if (epoch<0 and os.path.exists(results_name)) or (epoch>=0 and os.path.exists(results_name_file)):
        print("Run already saved")
        sys.exit()
    if saves:
        if epoch >= 0:
            print(results_name_file)
            results = open(results_name_file,'w')
        else:
            print(results_name)
            results = open(results_name,'w')

        out = 'epoch,avg_reward,avg_entropy,switches'

        for opt in range(num_options): out += ',option {} dur'.format(opt)
        for opt in range(num_options): out += ',option {} steps'.format(opt)
        for opt in range(num_options): out += ',option {} std'.format(opt)
        for opt in range(num_options): out += ',option {} term'.format(opt)
        for opt in range(num_options): out += ',option {} adv'.format(opt)
        out+='\n'
        results.write(out)
        # results.write('epoch,avg_reward,option 1 dur, option 2 dur, option 1 term, option 2 term\n')
        results.flush()

    if wsaves:
        first=True
        if not os.path.exists(results_name+'_weights'):
            os.makedirs(results_name+'_weights')
            first = False
        # while os.path.exists(dirname) and first:
        #     dirname += '0'

        # files = ['pposgd_simple.py','mlp_policy.py','run_mujoco.py']
        # for i in range(len(files)):
        #     src = os.path.expanduser('') + files[i]
        #     dest = os.path.expanduser('') + results_name+'_weights'
        #     shutil.copy2(src,dest)
    ###


    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    atarg_ent = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage with pseudo reward function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    ret_ent = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return Entropy
    diversity = tf.placeholder(dtype=tf.float32, shape=[None])

    # option = tf.placeholder(dtype=tf.int32, shape=[None])

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    # pdb.set_trace()
    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])

    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * (atarg_ent if deoc else atarg) # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * (atarg_ent if deoc else atarg) #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)  (Intra Option update)

    vf_loss = U.mean(tf.square(pi.vpred - ret)) ## Critic (Option critic)


    if deoc:
        vf_loss_ent = U.mean(tf.square(pi.vpred_ent - ret_ent))
        total_loss = pol_surr + pol_entpen + vf_loss + vf_loss_ent
        losses = [pol_surr, pol_entpen, vf_loss, vf_loss_ent, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "vf_ent_loss", "kl", "ent"]
    else:
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]


    if tdeoc:
        term_loss = -pi.tpred * diversity ## Termination loss fn
    else:
        term_loss = pi.tpred * term_adv

    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-20, 1.0))
    entropy = -tf.reduce_sum(pi.op_pi * log_pi, reduction_indices=1)

    op_loss = - tf.reduce_sum( log_pi[0][option[0]] * atarg  + entropy * 0.1 ) #Policy over options

    total_loss += op_loss
    
    var_list = pi.get_trainable_variables()
    term_list = var_list[8:10]

    lossandgrad = U.function([ob, ac, atarg, atarg_ent, ret, ret_ent, lrmult,option], losses + [U.flatgrad(total_loss, var_list)])
    termloss = U.function(([ob, option, term_adv, diversity] if tdeoc else [ob, option, term_adv]), [U.flatgrad(term_loss, var_list)]) # Since we will use a different step size.
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, atarg_ent, ret, ret_ent, lrmult, option], losses)


    U.initialize()
    adam.sync()


    saver = tf.train.Saver(max_to_keep=10000)
    dirname = results_name+'_weights/'

    if epoch >= 0:
        
        dirname = results_name+'_weights/'
        print("Loading weights from iteration: " + str(epoch))

        filename = dirname + '{}.ckpt'.format(epoch)
        saver.restore(U.get_session(),filename)
    ###    



    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    prewbuffer = deque(maxlen=100) # rolling buffer for episode pseudo rewards


    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_options=num_options,saves=saves,results=results,rewbuffer=rewbuffer,dc=dc)

    datas = [0 for _ in range(num_options)]

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam, deoc=deoc, tradeoff=tradeoff)



        opt_d = [0 for _ in range(num_options)]
        opt_steps =[] # Mean number of steps taken by an option before termination
        for i in range(num_options):
            dur = np.mean(seg['opt_dur'][i]) if len(seg['opt_dur'][i]) > 0 else 0.
            opt_steps.append(dur)

        std = []
        for i in range(num_options):
            logstd = np.mean(seg['logstds'][i]) if len(seg['logstds'][i]) > 0 else 0.
            std.append(np.exp(logstd))
        print("mean opt dur:", opt_steps)
        print("mean op pol:", np.mean(np.array(seg['optpol_p']),axis=0))
        print("mean term p:", np.mean(np.array(seg['term_p']),axis=0))
        print("mean value val:", np.mean(np.array(seg['value_val']),axis=0))
       

        ob, ac, opts, atarg, atarg_ent, tdlamret, tdlamret_ent, diversity = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["adv_ent"], seg["tdlamret"], seg["tdlamret_ent"], seg["prew"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        diversity = (diversity - diversity.mean()) / diversity.std()
        diversity = diversity*1e1 # if env.spec.id[:-3].lower()[:9] == "miniworld" else diversity*1e1
        atarg_ent = (atarg_ent if not deoc else (atarg_ent - atarg_ent.mean()) / atarg_ent.std())
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values



        if iters_so_far % 100 == 0 and wsaves:
            print("weights are saved...")
            filename =  dirname + '{}.ckpt'.format(iters_so_far)
            save_path = saver.save(U.get_session(),filename)


        min_batch= (160 if num_options<3 else 200) # Arbitrary
        t_advs = [[] for _ in range(num_options)]
        for opt in range(num_options):
            indices = np.where(opts==opt)[0]
            print("batch size:",indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                t_advs[opt].append(0.)
                continue


            ### This part is only necessasry when we use options. We proceed to these verifications in order not to discard any collected trajectories.
            if datas[opt] != 0:
                if (indices.size < min_batch and datas[opt].n > min_batch):
                    datas[opt] = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], atarg_ent=atarg_ent[indices],  vtarg=tdlamret[indices], vtarg_ent=tdlamret_ent[indices], diversity=diversity[indices]), shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif indices.size + datas[opt].n < min_batch:
                    # pdb.set_trace()
                    oldmap = datas[opt].data_map

                    cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                    cat_atarg_ent = np.concatenate((oldmap['atarg_ent'],atarg_ent[indices]))
                    cat_diversity = np.concatenate((oldmap['diversity'],diversity[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                    cat_vtarg_ent = np.concatenate((oldmap['vtarg_ent'],tdlamret_ent[indices]))
                    datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, atarg_ent=cat_atarg_ent, vtarg=cat_vtarg, vtarg_ent=cat_vtarg_ent, diversity=cat_diversity), shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (indices.size > min_batch and datas[opt].n < min_batch):

                    oldmap = datas[opt].data_map
                    cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                    cat_atarg_ent = np.concatenate((oldmap['atarg_ent'],atarg_ent[indices]))
                    cat_diversity = np.concatenate((oldmap['diversity'],diversity[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                    cat_vtarg_ent = np.concatenate((oldmap['vtarg_ent'],tdlamret_ent[indices]))
                    datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, atarg_ent=cat_atarg_ent, vtarg=cat_vtarg, vtarg_ent=cat_vtarg_ent, diversity=cat_diversity), shuffle=not pi.recurrent)

                if (indices.size > min_batch and datas[opt].n > min_batch):
                    datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], atarg_ent=atarg_ent[indices], vtarg=tdlamret[indices], vtarg_ent=tdlamret_ent[indices] , diversity=diversity[indices]), shuffle=not pi.recurrent)

            elif datas[opt] == 0:
                datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], atarg_ent=atarg_ent[indices], vtarg=tdlamret[indices], vtarg_ent=tdlamret_ent[indices], diversity=diversity[indices]) , shuffle=not pi.recurrent)
            ###



            optim_batchsize = optim_batchsize or ob.shape[0]
            optim_epochs = np.clip(np.int(10 * (indices.size / (timesteps_per_batch/num_options))),10,10) if num_options > 1 else optim_epochs
            print("optim epochs:", optim_epochs)
            logger.log("Optimizing...")


            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):

                    tadv,nodc_adv = pi.get_term_adv(batch["ob"],[opt])
                    tadv = tadv if num_options > 1 else np.zeros_like(tadv)
                    t_advs[opt].append(nodc_adv)

                    *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["atarg_ent"], batch["vtarg"], batch["vtarg_ent"], cur_lrmult, [opt])
                    termg = termloss(batch["ob"], [opt], tadv, batch["diversity"]) if tdeoc else termloss(batch["ob"], [opt], tadv)
                    adam.update(termg[0], term_mult*5e-7 * cur_lrmult)
                    adam.update(grads, lr_mult*optim_stepsize * cur_lrmult)
                    losses.append(newlosses)

        # Record 3d simulations
        if iters_so_far%50 == 0 and render:
            record_behavior(env, pi, iteration=iters_so_far, stochastic=True, num_opts=num_options, frames=2052, dirname=results_name)

        # Record Trajectories with option distinction
        if iters_so_far%5==0 and render :
            if hasattr(env,'NAME'):
                record_tmaze(env, pi, iteration=iters_so_far, stochastic=True, num_opts=num_options, frames=2052, dirname=results_name, epoch=epoch)
            elif env.spec.id[:-3].lower() == "miniworld-oneroom" or env.spec.id[:-3].lower() == "miniworld-tmaze":
                record_oneroom(env, pi, iteration=iters_so_far, stochastic=True, num_opts=num_options, frames=2052, dirname=results_name, epoch=epoch)


        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ent_rets"], seg["opt_switches"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews, prew, switches = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        prewbuffer.extend(prew)
        logger.record_tabular("OptSwitches", np.mean(switches))
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EntropyMean", np.mean(prewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()


        ### Book keeping
        if saves:
            out = "{},{},{},{}"
            for _ in range(num_options): out+=",{},{},{},{}"
            out+="\n"


            info = [iters_so_far, np.mean(rewbuffer), np.mean(prewbuffer), np.mean(seg["opt_switches"])]
            for i in range(num_options): info.append(opt_d[i])
            for i in range(num_options): info.append(opt_steps[i])
            for i in range(num_options): info.append(std[i])
            for i in range(num_options): info.append(np.mean(np.array(seg['term_p']),axis=0)[i])
            for i in range(num_options):
                info.append(np.mean(t_advs[i]))

            results.write(out.format(*info))
            results.flush()


def record_behavior(env, pi, iteration, stochastic=True, num_opts=1, frames=200, dirname = "",epoch=-1):
    episodes = 0
    counter = 0
    iteration = epoch if epoch>0 and iteration==0 else iteration
    dir_name = "Option_Behaviors/" + str(dirname) + "/" + str(iteration)
    dir_name_run = "Recorded_run/" + str(dirname) + "/" + str(iteration)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(dir_name_run):
        os.makedirs(dir_name_run)

    outs = [cv2.VideoWriter(os.path.join(dir_name, "Option_{}.avi".format(i)),cv2.VideoWriter_fourcc(*'XVID'), 50, (500,500)) for i in range(num_opts)]
    out_run = cv2.VideoWriter(os.path.join(dir_name_run, "Run.avi"),cv2.VideoWriter_fourcc(*'XVID'), 50, (500,500))
    ob = env.reset()
    option = pi.get_option(ob)
    for frame in range(frames):
        counter += 1
        if isinstance(pi, cnn.CnnPolicy):
            ac, _, _, _  = pi.act(stochastic, ob, option)
        else:
            ac, _, _, _ ,_ = pi.act(stochastic, ob, option)
        ob, rew, new, _ = env.step(ac)
        div = gen_pseudo_reward(pi, ob, num_opts, stochastic)
        term = pi.get_term([ob],[option])[0][0]

        if not hasattr(env,'NAME'):
            if env.spec.id[:-3].lower()[:9] == "miniworld":
                env.set_usertxt(option)
        frame = env.render(mode='rgb_array')
        frame = cv2.resize(frame, (500,500))
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame,'Option: {}'.format(option),(25,50), font, 0.5, (200,255,255), 1, cv2.LINE_AA)
        # frame = cv2.putText(frame,'Div: {0:.2f}'.format(div),(25,75), font, 0.5, (200,255,255), 1, cv2.LINE_AA)
        frame = cv2.putText(frame,'Term: {0:.2f}'.format(term),(25,100), font, 0.5, (200,255,255), 1, cv2.LINE_AA)
        outs[option].write(frame)
        out_run.write(frame)

        if term:
            option = pi.get_option(ob)
        if new:
            episodes += 1
            counter = 1
            ob = env.reset()
            option = pi.get_option(ob)
    print("episodes={}".format(episodes))
    [out.release() for out in outs]
    out_run.release()

def record_tmaze(env, pi, iteration, stochastic=True, num_opts=1, frames=200, dirname = "", epoch=-1):
    import matplotlib.pyplot as plt
    iteration = epoch if epoch>0 and iteration==0 else iteration
    episodes = 0
    counter = 0
    xs = np.arange(-.4,0.4,0.01)
    ys = np.arange(0.45,-0.2,-0.01)
    dir_name = "Trajectories/" + str(dirname) + "/" + str(iteration)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ob = env.reset()
    option = pi.get_option(ob)

    # Generate trajectories sample
    for _ in range(15):
        options = [[],[]]
        while True:
            ac, _, _, _ ,_ = pi.act(stochastic, ob, option)
            ob, rew, new, _ = env.step(ac)
            options[option].append(ob)
            term = pi.get_term([ob],[option])[0][0]

            if term:
                option = pi.get_option(ob)
            if new:
                episodes += 1
                counter = 1
                ob = env.reset()
                option = pi.get_option(ob)
                break

        # Plot Trajectories
        fig,ax = plt.subplots(1,1)
        ax.set_facecolor('white')
        ax.axvspan(-0.1, 0.1, facecolor='black',zorder=1)
        ax.axhspan(0.2,0.45,  facecolor='black',zorder=1)
        if options[0]:
            ax.scatter(np.array(options[0])[:,0],np.array(options[0])[:,1],s=15,c='red',zorder=2)#, 'r*', zorder=2)     #color='xkcd:red',linestyle=':', zorder=2)
        if options[1]:
            ax.scatter(np.array(options[1])[:,0],np.array(options[1])[:,1],s=15,c='yellow',zorder=2)#, 'y*', zorder=2)  #color='xkcd:fuchsia',linestyle=':', zorder=2)

        # Draw maze
        # ax.plot(np.arange(-0.1,0.1,0.001), np.full((200,),-0.2) ,c='black', linewidth=4)
        # ax.plot(np.full((400,),0.1), np.arange(-0.2,0.2,0.001),c='black', linewidth=4)
        # ax.plot(np.full((400,),-0.1), np.arange(-0.2,0.2,0.001),c='black', linewidth=4)
        #
        # ax.plot(np.arange(0.1,0.4,0.001), np.full((301,),0.2) ,c='black', linewidth=4)
        # ax.plot(np.arange(-0.4,-0.1,0.001), np.full((301,),0.2) ,c='black', linewidth=4)
        #
        # ax.plot(np.full((650,),0.4), np.arange(-0.2,0.45,0.001),c='black', linewidth=4)
        # ax.plot(np.full((650,),-0.4), np.arange(-0.2,0.45,0.001),c='black', linewidth=4)
        #
        # ax.plot(np.arange(-0.4,0.4,0.001), np.full((800,),0.45),c='black' , linewidth=4)

        ax.scatter(0.3,0.3, s=1000, c='lime',zorder=2)
        ax.scatter(-0.3,0.3, s=1000, c='lime',zorder=2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xmin=-.4,xmax=.4)
        ax.set_ylim(ymin=-.2,ymax=.45)
        # ax.set_facecolor('white')
        # ax.axvspan(-0.1, 0.1, facecolor='black')
        # ax.axhspan(0.2,0.45,  facecolor='black')
        plt.savefig(os.path.join(dir_name, "Ep{}.png".format(episodes)),bbox_inches='tight')
        plt.clf();plt.close()


    #######################  Plot teriminations ####################################
    import matplotlib.patches as patches
    dir_name = "Terminations/" + str(dirname) + "/" + str(iteration)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    vinput= []
    for y in ys:
        for x in xs:
            vinput.append([x,y])
    vinput = np.array(vinput).squeeze()
    terms = [[],[]]
    for op in range(len(options)):
        for i in range(len(vinput)):
            terms[op].append(pi.get_tpred([vinput[i]],[op])[0][0])
    # terms = np.array(terms)

    for i in range(len(options)):
        fig,ax = plt.subplots(1,1)
        # for a in ax:
        #     a.axvspan(-0.1, 0.1, facecolor='black',zorder=1)
        #     a.axhspan(0.2,0.45,  facecolor='black',zorder=1)
        heatmap_op1 = ax.imshow(np.array(terms[i]).reshape(len(ys),len(xs)), extent=[-.4,.4,-.2,.45], interpolation='nearest', alpha=1.,zorder=1)
        # heatmap_op2 = ax.imshow(np.array(terms[1]).reshape(len(ys),len(xs)), extent=[-.4,.4,-.2,.45], interpolation='nearest', alpha=1.,zorder=1)
        # for a in ax:
        ax.scatter(0.3,0.3, s=1000, c='lime',zorder=3)
        ax.scatter(-0.3,0.3, s=1000, c='lime',zorder=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(patches.Rectangle((-0.398,-0.2),0.3,0.4, color='white',zorder=4))
        ax.add_patch(patches.Rectangle((0.1,-0.2),0.3,0.4, color='white',zorder=4))
        plt.colorbar(heatmap_op1, ax=ax)

        plt.savefig(os.path.join(dir_name, "Op{}.png".format(i)),bbox_inches='tight')
        plt.clf();plt.close()

    ################# Plot terminations during trajectories ##########################
    dir_name = "Terminations_Trajectories/" + str(dirname) + "/" + str(iteration)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for _ in range(5):
        options = [[],[]]
        while True:
            ac, _, _, _ ,_ = pi.act(stochastic, ob, option)
            ob, rew, new, _ = env.step(ac)
            # options[option].append(ob)
            term = pi.get_term([ob],[option])[0][0]

            if term:
                options[option].append(ob)
                option = pi.get_option(ob)
            if new:
                episodes += 1
                counter = 1
                ob = env.reset()
                option = pi.get_option(ob)
                break

        # Plot Trajectories
        fig,ax = plt.subplots(1,1)
        ax.set_facecolor('white')
        ax.axvspan(-0.1, 0.1, facecolor='black',zorder=1)
        ax.axhspan(0.2,0.45,  facecolor='black',zorder=1)
        if options[0]:
            ax.scatter(np.array(options[0])[:,0],np.array(options[0])[:,1],s=15,c='red',zorder=2)#, 'r*', zorder=2)     #color='xkcd:red',linestyle=':', zorder=2)
        if options[1]:
            ax.scatter(np.array(options[1])[:,0],np.array(options[1])[:,1],s=15,c='yellow',zorder=2)#, 'y*', zorder=2)  #color='xkcd:fuchsia',linestyle=':', zorder=2)

        ax.scatter(0.3,0.3, s=1000, c='lime',zorder=2)
        ax.scatter(-0.3,0.3, s=1000, c='lime',zorder=2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(xmin=-.4,xmax=.4)
        ax.set_ylim(ymin=-.2,ymax=.45)
        # ax.set_facecolor('white')
        # ax.axvspan(-0.1, 0.1, facecolor='black')
        # ax.axhspan(0.2,0.45,  facecolor='black')
        plt.savefig(os.path.join(dir_name, "Ep{}.png".format(episodes)),bbox_inches='tight')
        plt.clf();plt.close()

def record_oneroom(env, pi, iteration, stochastic=True, num_opts=1, frames=200, dirname = "", epoch=-1):
    import matplotlib.pyplot as plt
    episodes = 0
    counter = 0
    iteration = epoch if epoch>0 and iteration==0 else iteration
    xs = np.arange(-.4,0.4,0.01)
    ys = np.arange(0.45,-0.2,-0.01)
    dir_name = "Trajectories/" + str(dirname) + "/" + str(iteration)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    ob = env.reset()
    option = pi.get_option(ob)

    # Generate trajectories sample
    for _ in range(10):
        options = [[],[]]
        frame2 = env.render_top_view()
        frame2 = cv2.resize(frame2, (520,394))
        fig,ax = plt.subplots(1,1)
        goal_pos = [env.box.pos[0], env.box.pos[2]]
        start_pos = [env.agent.pos[0], env.agent.pos[2]]
        # ax.imshow(frame2, zorder=1)
        while True:
            ac, _, _, _  = pi.act(stochastic, ob, option)
            ob, rew, new, _ = env.step(ac)
            pos = [env.agent.pos[0], env.agent.pos[2]]
            options[option].append(pos)
            term = pi.get_term([ob],[option])[0][0]

            if term:
                option = pi.get_option(ob)
            if new:
                episodes += 1
                counter = 1
                ob = env.reset()
                option = pi.get_option(ob)
                break

        if options[0]:
            ax.scatter(np.array(options[0])[:,0],np.array(options[0])[:,1], marker='o',zorder=2, c='red')#, 'r*', zorder=2)     #color='xkcd:red',linestyle=':', zorder=2)
        if options[1]:
            ax.scatter(np.array(options[1])[:,0],np.array(options[1])[:,1], marker='o',zorder=2, c='yellow')#, 'y*', zorder=2)  #color='xkcd:fuchsia',linestyle=':', zorder=2)
        ax.scatter(goal_pos[0], goal_pos[1], s=100, marker='s', c='green')
        ax.scatter(start_pos[0], start_pos[1], s=60, marker='X', c='blue')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        plt.savefig(os.path.join(dir_name, "Ep{}.png".format(episodes)),bbox_inches='tight')
        plt.clf();plt.close()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

