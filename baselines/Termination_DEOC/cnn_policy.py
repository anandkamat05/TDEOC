import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np
import pdb
import datetime


def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w[option[0]])
    if bias:
        b = tf.get_variable(name + "/b", [num_options,size], initializer=tf.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.compat.v1.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.compat.v1.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, num_options=2,dc=0, kind='small'):
        assert isinstance(ob_space, gym.spaces.Box)

        self.dc = dc
        self.num_options = num_options
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        option =  U.get_placeholder(name="option", dtype=tf.int32, shape=[None])

        x = ob / 255.0
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, 'lin', U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError


        # Network to compute value function and termination probabilities
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = x
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = dense3D2(last_out, 1, "vffinal", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:,0]

        self.vpred_ent = dense3D2(last_out, 1, "vffinal_ent", option, num_options=num_options, weight_init=U.normc_initializer(1.0))[:,0]

        self.tpred = tf.nn.sigmoid(dense3D2(tf.stop_gradient(last_out), 1, "termhead", option, num_options=num_options, weight_init=U.normc_initializer(1.0)))[:,0]
        termination_sample = tf.greater(self.tpred, tf.random_uniform(shape=tf.shape(self.tpred),maxval=1.))


        # Network to compute policy over options and intra_option policies
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        # if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Discrete):
        #     mean = dense3D2(last_out, pdtype.param_shape()[0]//2, "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.01))
        #     logstd = tf.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
        #     pdparam = U.concatenate([mean, mean * 0.0 + logstd[option[0]]], axis=1)
        # else:
        pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.op_pi = tf.nn.softmax(U.dense(tf.stop_gradient(last_out), num_options, "OPfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))

        self.pd = pdtype.pdfromflat(pdparam)


        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, option], [ac, self.vpred, self.vpred_ent, last_out])
        self._get_logits = U.function([stochastic, ob, option], [self.pd.logits] )


        self._get_v = U.function([ob, option], [self.vpred])
        self._get_v_ent = U.function([ob, option], [self.vpred_ent])  # Entropy value estimate
        self.get_term = U.function([ob, option], [termination_sample])
        self.get_tpred = U.function([ob, option], [self.tpred])
        self.get_vpred = U.function([ob, option], [self.vpred])
        self.get_vpred_ent = U.function([ob, option], [self.vpred_ent]) # Entropy value estimate
        self._get_op = U.function([ob], [self.op_pi])


    def get_logits(self, stochastic, ob, option):
        logits = self._get_logits(stochastic, ob[None], [option])
        return logits[0]

    def act(self, stochastic, ob, option):
        ac1, vpred1, vpred_ent1, feats =  self._act(stochastic, ob[None], [option])
        return ac1[0], vpred1[0], vpred_ent1[0], feats[0]

    def get_option(self,ob):
        op_prob = self._get_op([ob])[0][0]
        return np.random.choice(range(len(op_prob)), p=op_prob)


    def get_term_adv(self, ob, curr_opt):
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        op_prob = self._get_op(ob)[0].transpose()
        return (vals[curr_opt[0]] - np.sum((op_prob*vals),axis=0) + self.dc),  ( vals[curr_opt[0]] - np.sum((op_prob*vals),axis=0) )

    def get_val(self, ob):
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        op_prob = self._get_op(ob)[0].transpose()

        return np.sum((op_prob*vals),axis=0)


    def get_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

