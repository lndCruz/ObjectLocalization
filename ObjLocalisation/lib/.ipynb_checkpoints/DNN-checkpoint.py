import numpy as np
import os
import sys
import tensorflow as tf
import cv2
from os.path import join

"""
Code from limit memory GPU
"""
#"""
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
#"""

# Agent Actions: 0 (right), 1 (down), 2 (scale up), 3 (aspect ratio up), 4 (left), 5 (up), 6 (scale down), 7 (aspect ratio down), 8 (split horizontal), 9 (split vetical), and 10 (termination) are valid actions

VALID_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class StateProcessor():
    """
    Processes raw images. Resizes it and converts it to grayscale.
    """
    def __init__(self):

        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[84, 84, 3], dtype=tf.uint8) #Essa linha para quando for RGB
            #self.input_state = tf.placeholder(shape=[224, 224, 1], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state) #Essa linha e a de baixo eh para quando a entrada da imagem for RGB
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #self.output = tf.image.resize_images(self.input_state, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [X, Y, Z] image RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        
        #state = np.zeros((224, 224, 1)) #Essa linha foi inserida para quando imagem for grayscale
        #quando for RGB, basta remover a linha de cima.

        #state = np.zeros((84, 84, 3))
        #print("lllll {}".format(state.shape))
        
        return sess.run(self.output, { self.input_state: state })



class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                # Old API: tf.train.SummaryWriter. It might be needed to use on GPU cluster
                self.summary_writer = tf.summary.FileWriter(summary_dir) 

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X") #RGB
        #self.X_pl = tf.placeholder(shape=[None, 224, 224, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        self.keep_prob = tf.placeholder(tf.float32)

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers 
        # To change the neural network architecture this part should be modified
        # Note if you wish to change the architecture you need to modify visulize_layers function as well 
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        flattened = tf.nn.dropout(flattened, self.keep_prob)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())
        # Summaries for Tensorboard
        
        # Old APIs for using on GPU cluster
        '''tf.scalar_summary("loss", self.loss, collections=['summ'])
        tf.histogram_summary("loss_hist", self.losses, collections=['summ'])
        tf.histogram_summary("q_values_hist", self.predictions, collections=['summ'])
        tf.scalar_summary("max_q_value", tf.reduce_max(self.predictions), collections=['summ'])
        self.summaries = tf.merge_all_summaries(key='summ')'''
        
        self.summaries = tf.summary.merge([ 
            tf.summary.scalar("loss", self.loss), 
            tf.summary.histogram("loss_hist", self.losses), 
            tf.summary.histogram("q_values_hist", self.predictions), 
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s, keep_prob = 1):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size , 80, 80, 4]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """

        return sess.run(self.predictions, { self.X_pl: s, self.keep_prob: keep_prob })
    
    def visulize_layers(self, sess, s, layer):

        """
        Returns Conv layers filters for visulazation purposes

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size , 80, 80, 4]
          layer: Layer number which is desired to visualize
        
        Returns:
          A conv layer of shape [4, 84, 84, filter_num]
        """ 
        conv1, conv2, conv3 = sess.run([self.conv1, self.conv2, self.conv3], { self.X_pl: s })
        if layer == '1':
            return conv1
        elif layer =='2':
            return conv2
        elif layer == '3':
            return conv3
        

        

    def update(self, sess, s, a, y, keep_prob = 0.5):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 84, 84, 4]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]
          keep_prob: Dropout probability of keeping neurons

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a, self.keep_prob: keep_prob }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            # Old API for using on cluster
            # self.summary_writer.add_summary(summaries, global_step.eval())
            self.summary_writer.add_summary(summaries, global_step)
        return loss





class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """
    
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.  
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)





def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
      estimator: An estimator that returns q values for a given state
      nA: Number of actions in the environment.
    Returns:
       A policy for a given estimator

    """
    def policy_fn(sess, observation, epsilon):
        """
        Predicts Q values and gives probability distribution over actions
        
        Args:
          sess: Tensorflow session object
          observation: State input of shape [84, 84, 4]
          epsilon: Probability of taking actions rendomly

        Returns:
          the probabilities for every action in the form of a numpy array of length nA.

        """
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A, q_values
    return policy_fn





