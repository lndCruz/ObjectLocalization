import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
from PIL import Image

from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser


def DQL_testing(num_episodes, category, model_name):
    """
    Evaluates a model on testing set.
    Args:
       num_episodes: Number of episodes that the agect can interact with an image
       category: The category that is going to be used for evaluation
       model_name: The model name that is going to be evaluated
    Returns:
       Mean precision for the given category over test set
    """

    # Checks whether records are availible
    destination = "../data/"
    if not (os.path.isfile(destination+"test_input.npz") or os.path.isfile(destination+"test_target.npz")):
        print("Files are not ready!!!")
        return 0
    else:
        print("Records are already prepared!!!")



    # Initiates Tensorflow graph
    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("../experiments/{}".format(model_name))

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)

    # State processor
    state_processor = StateProcessor()


    with tf.Session() as sess:


        # For 'system/' summaries, usefull to check if currrent process looks healthy
        current_process = psutil.Process()

        # Create directories for checkpoints and summaries
        checkpoint_dir = os.path.join(experiment_dir, "bestModel")
        checkpoint_path = os.path.join(checkpoint_dir, "model")

        # Initiates a saver and loads previous saved model if one was found
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        # Get the current time step
        total_t = sess.run(tf.contrib.framework.get_global_step())


        # The policy we're following
        policy = make_epsilon_greedy_policy(
            q_estimator,
            len(VALID_ACTIONS))

        precisions = []

        contImage = 0
        
        for indx,tmp in enumerate(extractData(category, "test", 32)):

            
            # Unpacking image and ground truth 
            img=tmp[0]
            target=tmp[1]
            succ = 0

            # Creates an object localizer instance
            im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
            env = ObjLocaliser(np.array(im2),target)
            print ("Image{} is being loaded: {}".format(indx, img['image_filename']))

            # Num of episodes that Agent can interact with an input image 
            for i_episode in range(num_episodes):


                # Reset the environment
                env.Reset(np.array(im2))
                state = env.wrapping()
                state = state_processor.process(sess, state)
                state = np.stack([state] * 4, axis=2)

                t=0
                action = 0

                # The agent searches in an image until terminatin action is used or the agent reaches threshold 50 actions
                while (action != 10) and (t < 50):

                    # Choosing action based on epsilon-greedy with probability 0.8
                    action_probs, qs = policy(sess, state, 0.2)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                    # Takes action and observes new state and reward
                    reward = env.takingActions(VALID_ACTIONS[action])
                    next_state = env.wrapping()
                    
                    try:
                    
                        if reward == 3:
                            succ += 1

                        # Processing the new state
                        next_state = state_processor.process(sess, next_state)
                        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                        state = next_state

                        t += 1
                        
                    except:
                        pass

                print ("number of actions for step {} is: {}".format(i_episode, t))

            contImage += 1
            precisions.append(float(succ)/num_episodes)
            print ("image {} precision: {}".format(img['image_filename'], precisions[-1]))



    print ("num of images:{}".format(len(precisions)))
    
    print ("num Total of images:{}".format(contImage))

    print ("mean precision: {}".format(np.mean(precisions)))

    return np.mean(precisions)



