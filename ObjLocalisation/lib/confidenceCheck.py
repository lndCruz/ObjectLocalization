import numpy as np
import math

lossMax = 0
    
def calcConfidence(loss):
    
    """
    Calcula quão confiante o agente está na sugestão do DQN.
    Quando a confiança no DQN é baixa, o usuario aconselha qual acao deve ser realizada,
    caso contrário, o DQN aplica ação escolhida pela rede.
    
    Calculo extraido do artigo:
    Explore, Exploit or Listen: Combining Human Feedback and Policy Model to Speed up Deep Reinforcement Learning in 3D Worlds

    Args:
      loss: valor do loss da DQN


    Returns:
      O valor da confiança
    """
    
    if(loss > lossMax):
        lossMax = loss
        
    #print("loss"+str(loss)+ ""+"lossMax" +str(lossMax))
    
    x = math.sqrt(loss/lossMax)
    y = np.log(x)
                            
    resultado = -1/y-1
    
    print("CHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMMMMMMMMMMMMMMMMMMMMMMOOUUUUUU " + str(resultado))
    #return resultado