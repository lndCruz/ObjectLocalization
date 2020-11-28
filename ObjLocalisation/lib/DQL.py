import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple
from readingFileEfficiently import *
import VOC2012_npz_files_writter
from DNN import *
from Agent import ObjLocaliser
import cv2


#LeoAqui
import math
from confidenceCheck import calcConfidence


"""
Code from limit memory GPU
"""
#"""
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
#"""


def preparedataset():
    """
    Downloads VOC 2012 dataset and prepares it to be used for training and Testing. 
    """

    # Path to the dataset annotation
    xml_path = "../GE_MAMMO/Annotations/nipple/*.xml"
    #xml_path = "../CXR_Cardiomegaly/Annotations/cardiomegaly/*.xml"
    #xml_path = "../CXR_Mass/Annotations/mass/*.xml"
    
    #xml_path = "../CXR_Infiltration/Annotations/infiltrate/*.xml"
    #xml_path = "../VOC2012/Annotations/*.xml"
    #xml_path = "../DIR/Annotations/*.xml"


    # Path to the prepared data
    destination = "../data/"


    # Checks whether the data is already prepared
    if not (os.path.isfile(destination+"test_input.npz") or os.path.isfile(destination+"test_target.npz")):

	# Checks whether the dataset is already downloaded
        #leo aqui:comentei essas linhas pq eu ja tinha o dataset
        #if not os.path.isfile("../VOCtrainval_11-May-2012.tar"):

        #	print "downloading VOC2012 dataset to ../pascal-voc-2012.zip ..."
        #	os.system("wget -P ../ http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar")
        #	print "download finished."

        # Unziping the dataset
        if not os.path.isdir("../GE_MAMMO"):

            print ("Unziping the files ...")
            os.system("tar xf ../VOCtrainval_11-May-2012.tar -C ../")
            os.system("cp -r ../VOCdevkit/* ../")
            os.system("rm -r ../VOCdevkit")
    
        # Writting the dataset to .npz files
        os.system("mkdir ../data")
        # Splits dataset to 80% for training and 20% validation
        # This cell reads VOC 2012 dataset and save them in .npz files for future
        #VOC2012_npz_files_writter.writting_files(xml_path, destination, percentage=0)
        VOC2012_npz_files_writter.writting_files(xml_path, destination)
        print("Files are ready!!!")
        
    else:
        print("Records are already prepared!!!")


def evaluate(tmp, state_processor, policy, sess, num_of_proposal=15):
    """
    Evaluates a given network on an image

    Args:
      tmp: A tuple of [image, target]
      state_processor: An instance of StateProcessor class
      policy: An instance of make_epsilon_greedy_policy function
      sess: Tensorflow session object
      num_of_proposal: Number of proposals that are used for evaluation

    Returns:
      Mean precision for the input image
    """


    # Unpacking input image and its ground truth
    img=tmp[0]
    target=tmp[1]
    succ = 0

    FN = 0 #para calcular falso negativo da MAP
    
    # Creates an object localizer instance
    #im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image']) PARA RGB
    im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
    env = ObjLocaliser(np.array(im2),target)


    # Num of episodes that Agent can interact with an input image 
    for i_episode in range(num_of_proposal):


        # Reset the environment
        env.Reset(np.array(im2))
        state = env.wrapping()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 3, axis=2)

        t=0
        action = 0

        # The agent searches in an image until terminatin action is used or the agent reaches threshold 50 actions
        while (action != 8) and (t < 700):


            # Choosing action based on epsilon-greedy with probability 0.8
            action_probs, qs = policy(sess, state, 0.2)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            # Taking action in environment and recieving reward
            reward = env.takingActions(VALID_ACTIONS[action])
        # If an object is successfuly localized increase counter
            if reward == 3:
                succ += 1
                
            if reward == -3:
                FN += 1
            # Observing next state
            next_state = env.wrapping()
            #leo aqui
            #if next_state is None:
               
            #fim do leo
            try:
                next_state = state_processor.process(sess, next_state)
                next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                state = next_state

                t += 1
                
            except:
                t+=1

    #return (float(succ)/num_of_proposal)
    return (float(succ)/float(succ + FN))
   

#Tentativa de Implementando aqui o artigo teaching on budget
# Teaching on a Budget in Multi-Agent Deep Reinforcement Learning
    
    

def DQL(num_episodes,
     replay_memory_size,
     replay_memory_init_size,
     update_target_estimator_every,
     discount_factor,
     epsilon_start,
     epsilon_end,
     epsilon_decay_steps,
     category,
     model_name):



    """
    Builds and trains deep Q-network

    Args:
       num_episodes: Number of episodes that the agect can interact with an image
       replay_memory_size: Number of the most recent experiences that would be stored
       replay_memory_init_size: Number of experiences to initialize replay memory
       update_target_estimator_every: Number of steps after which estimator parameters are copied to target network
       discount_factor: Discount factor
       epsilon_start: Epsilon decay schedule start point
       epsilon_end: Epsilon decay schedule end point
       epsilon_decay_steps: Epsilon decay step rate
       category: Indicating the categories are going to be used for training
       model_name: The trained model would be saved with this name
    """



    # Downloads and prepares dataset
    preparedataset()

    # Initiates Tensorflow graph
    tf.reset_default_graph()

    # Where checkpoints and graphs are saved
    experiment_dir = os.path.abspath("../experiments/{}".format(model_name))

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
    target_estimator = Estimator(scope="target_q")

    # State processor
    state_processor = StateProcessor()




    with tf.Session() as sess:
        

        # Initializes the network weights
        sess.run(tf.initialize_all_variables())
        Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        # The replay memory
        replay_memory = []

        # Make model copier object
        estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

        # For 'system/' summaries, usefull to check if currrent process looks healthy
        current_process = psutil.Process()

        # Create directories for checkpoints and summaries
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "model")
        report_path = os.path.join(experiment_dir, "report")
        best_model_dir = os.path.join(experiment_dir, "bestModel")
        best_model_path = os.path.join(best_model_dir, "model")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(report_path):
            os.makedirs(report_path)
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        f = open(report_path+"/log.txt", 'w')

        # Initiates a saver and loads previous saved model if one was found
        saver = tf.train.Saver()
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        # Get the current time step
        total_t = sess.run(tf.contrib.framework.get_global_step())

        # The epsilon decay schedule
        epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # The policy we're following
        policy = make_epsilon_greedy_policy(
            q_estimator,
            len(VALID_ACTIONS))

        # Initiates counters
        episode_counter = 0
        best_pre = 0
        eval_pre = []
        eval_set = []
        batch_size=32
        done = False
        num_located = 0
        contImage = 0
        qtd_imageDetected = 0 #para saber quantas imagens foram detectadas corretamente
        lossMax=0
        resultado = 0
        cont_image = 0

        # Loads images from dataset
        for indx,tmp in enumerate(extractData(category, "train", batch_size)):
                        
            #Contador para ver quantas imagens foram lidas
            contImage += 1
            
            if contImage >= 31:
                break;
            
            # Unpacking image and ground truth 
            img=tmp[0]
            target=tmp[1]
            
            
            #ESSE TRECH EH PARA IMPRIMIR IMAGEM LOGO DE CARA
            #im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
            #teste = ObjLocaliser(np.array(im2),target)
                
            #CODIGO PARA VER ACOES
            #teste.drawActions(img['image_filename'])
            

            # The first 100 images are used for evaluation
            if len(eval_set) < 10:
                print ("Populating evaluation set...")
                eval_set.append(tmp)

            else:
                # Every 20 images the neural network is evaluated
                if indx%9 == 0:
                    print ("Evaluation started ...")
                    print("Every 20 images the neural network is evaluated")
                    for tmp2 in eval_set:
                        eval_pre.append(evaluate(tmp2, state_processor, policy, sess))
                        if len(eval_pre) > 9:
                             
                            # Saves the result of evaluation with Tensorboard
                            print ("Evaluation mean precision: {}".format(np.mean(eval_pre)))
                            f.write("Evaluation mean precision: {}\n".format(np.mean(eval_pre)))
                            episode_summary = tf.Summary()
                            episode_summary.value.add(simple_value=np.mean(eval_pre), tag="episode/eval_acc")
                            q_estimator.summary_writer.add_summary(episode_summary, episode_counter)
                            q_estimator.summary_writer.flush()

                            # If the achieved result is better than the previous results current state of the model is saved
                            if np.mean(eval_pre) > best_pre:
                                print ("Best model changed with mean precision: {}".format(np.mean(eval_pre)))
                                f.write("Best model changed with mean precision: {}\n".format(np.mean(eval_pre)))
                                best_pre = np.mean(eval_pre)
                                saver.save(tf.get_default_session(), best_model_path)
                            eval_pre = []

                # Creates an object localizer instance
                #im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image']) PARA RGB
                im2 = Image.frombytes("RGB",(img['image_width'],img['image_height']),img['image'])
                env = ObjLocaliser(np.array(im2),target)
                
                
                print ("Image{} is being loaded: {}".format(indx, img['image_filename']))
                f.write("Image{} is being loaded: {}".format(indx, img['image_filename']))

                # Populates the replay memory with initial experiences
                if len(replay_memory) < replay_memory_init_size:

                    print("Populating replay memory...\n")
                    
                    # Reads and processes the current state
                    env.Reset(np.array(im2))
                    state = env.wrapping()
                    state = state_processor.process(sess, state)
                    state = np.stack([state] * 3, axis=2)

                    # Populating replay memory with the minimum threshold 
                    for i in range(replay_memory_init_size):

                        print("############################################")
                        print("##########")
                        print("##########")
                        print("Tamanho do threshold replay memory " + str(i))
                        print("##########")
                        print("##########")
                        print("############################################")
                        
                        #Criando decisao para permitir que o usuario insira 100 anotações
                        #Primeiro o usuario precisa olhar a imagem salva no DIR anim
                        if i > 100:
                            if i == 1:#Essa verificacao eh para garantir na primeira vez o reset antes do input do usuario
                                env.Reset(np.array(im2))
                                
                            env.drawActions(img['image_filename'])
                            print("Input your advice:")
                            action = int(input())
                            
                        else:
                        
                            # Epsilon for this time step 
                            action_probs, _ = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
                            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                        # Takes action and observes new state and reward
                        reward = env.takingActions(VALID_ACTIONS[action])
                        next_state = env.wrapping()
                        
                        
                        # Checks whether termination action is taken
                        if action == 8:
                            done = True
                        else:
                            done = False
                        try:
                            # Processing the new state
                            next_state = state_processor.process(sess, next_state)
                            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
                            replay_memory.append(Transition(state, action, reward, next_state, done))
                            state = next_state

                            if done:
                                env.Reset(np.array(im2))
                                state = env.wrapping()
                                state = state_processor.process(sess, state)
                                state = np.stack([state] * 3, axis=2)
                            else:
                                state = next_state
                        except:
                            pass

                vet = [] # define um vetor vazio para armazenar as acoes
                
                # Num of episodes that Agent can interact with an input image 
                for i_episode in range(num_episodes):
                    
                    actionAdvice = 0 #LeoAqui
                    
                    #Variavel que limite a quantidade de vezes que o usuario vai poder dar conselhos
                    budget = 5
                    
                    # Save the current checkpoint
                    saver.save(tf.get_default_session(), checkpoint_path)

                    # Reset the environment
                    env.Reset(np.array(im2))
                    state = env.wrapping()
                    state = state_processor.process(sess, state)
                    state = np.stack([state] * 3, axis=2)
                    loss = 0
                    t=0
                    action = 0
                    e = 0
                    r = 0

                    # The agent searches in an image until terminatin action is used or the agent reaches threshold 50 actions
                    while (action != 8) and (t < 700):
                        
                        # Epsilon for this time step
                        epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

                        # Maybe update the target estimator
                        if total_t % update_target_estimator_every == 0:
                            estimator_copy.make(sess)
                            print("\nCopied model parameters to target network.")

                                
                        #Realizando a verificao da confidence
                        #if resultado <= 1.0:

                            # Take a step
                            #action_probs, qs = policy(sess, state, epsilon)
                            #action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                            
                        #else:
                            #env.drawActions(img['image_filename'])
                            #print("Input your advice:")
                            #action = int(input()) 
                        '''
                        if contImage <= 30:
                            print("Qtd de imagens já lidas {}".format(contImage))
                            
                            if (i_episode + 1) == 1:

                                print("valor que jah tem na matriz " + str(vet))
                                
                                env.drawActions(img['image_filename'])
                                print("Input your advice:")
                                action = int(input())
                                
                                
                            else:
                                print("Acoes que estao na matriz" + str(vet))
                                action = vet[actionAdvice]
                                print("Acao escolhida " + str(action))
                                actionAdvice += 1
                        
                        #if action == 10:
                            #env.drawActions(img['image_filename'])
                            #print("confirma a acao? ")
                            #action = int(input())
                        
                        else:
                            action_probs, qs = policy(sess, state, epsilon)
                            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                        
                        vet.append(action)
                        '''                       
                        
                        
                        # Early advising
                        '''
                        #if contImage <= 15:
                        if budget > 0:
                            
                            budget = budget - 1
                            
                            mu = loss
                            

                            print("kkkkkkkkkkkkkkkkkkkkkkkkkk {}".format(mu))  

                            if mu >= 1.2:
                                env.drawActions(img['image_filename'])
                                print("Input your advice: ")
                                action = int(input())
                            else:
                                action_probs, qs = policy(sess, state, epsilon)
                                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                        else:
                            action_probs, qs = policy(sess, state, epsilon)
                            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

                        '''
                        action_probs, qs = policy(sess, state, epsilon)
                        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                        
                        # Takes action and observes new state and its reward
                        reward = env.takingActions(VALID_ACTIONS[action])
                        
                        next_state = env.wrapping()
                        
                        if action == 8:
                            done = True
                        else:
                            done = False
                        
                        #The origin code not have try/except. I put because I had an error when image shape was 0
                        try:
                            # Processing the new state
                            next_state = state_processor.process(sess, next_state)
                            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

                            # If our replay memory is full, pop the first element
                            if len(replay_memory) == replay_memory_size:
                                replay_memory.pop(0)

                            # Save transition to replay memory
                            replay_memory.append(Transition(state, action, reward, next_state, done))


                            # Sample a minibatch from the replay memory
                            samples = random.sample(replay_memory, batch_size)
                            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

                            # Calculate q values and targets
                            q_values_next = target_estimator.predict(sess, next_states_batch)
                            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * np.amax(q_values_next, axis=1)

                            # Perform gradient descent update
                            states_batch = np.array(states_batch)
                            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)
                            
                            #TESTE LEO
                            #
                            #
                            #calcConfidence(loss)
                            #
                            #
                            ######
                            if(loss > lossMax):
                                lossMax = loss
                             
                            #print("loss"+str(loss)+ ""+"lossMax" +str(lossMax))
                            #t = math.sqrt(loss/3.0)-1
                            x = math.sqrt(loss/lossMax)
                            y = np.log(x)
                            
                            resultado = -1/y-1

                            #print("CHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMMMMMMMMMMMMMMMMMMMMMMOOUUUUUU " + str(resultado))
                            
                            print("Step {} ({}) @ Episode {}/{}, action {}, reward {},loss: {}".format(t, total_t, i_episode + 1, num_episodes, action, reward, loss))
                            f.write("Step {} ({}) @ Episode {}/{}, action {}, reward {},loss: {}\n".format(t, total_t, i_episode + 1, num_episodes, action, reward, loss))


                            # Counting number of correct localized objects
                            if reward == 3:
                                num_located += 1
                                
                            state = next_state
                            t += 1
                            total_t += 1
                            e = e + loss
                            r = r + reward
                            
                        except:
                            pass

                    episode_counter += 1

                    # Add summaries to tensorboard
                    episode_summary = tf.Summary()
                    episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
                    episode_summary.value.add(simple_value=r, tag="episode/reward")
                    episode_summary.value.add(simple_value=t, tag="episode/length")
                    episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
                    episode_summary.value.add(simple_value=current_process.memory_percent(), tag="system/v_memeory_usage_percent")
                    q_estimator.summary_writer.add_summary(episode_summary, episode_counter)
                    q_estimator.summary_writer.flush()

                    print("Episode Reward: {} Episode Length: {}".format(r, t))
                    f.write("Episode Reward: {} Episode Length: {}".format(r, t))
                    
                    #SALVANDO IMAGENS NO DIR ANIM COM BBOX AND GT
                    #env.drawActions(img['image_filename'])
                    
                    if reward == 3:
                        cont_image += 1
                    
                #Coloquei essas verificaceos para ver quantas imagens o agente detectou corretamente
                if cont_image >= 1:
                    qtd_imageDetected += 1
                    cont_image = 0;
                    
        
        print('total de imagens TREINADAS {}'.format(contImage))
        f.write("total de imagens TREINADAS {}".format(contImage))
        print('Categoria das imagens {}'.format(category))
        f.write("Categoria das imagens {}".format(category))
        
    f.close()
    print ("quantidade de imagens que o agente conseguiu detectar corretamente: {}".format(qtd_imageDetected))



