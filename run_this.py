from env import Env
from myDQN import DeepQNetwork
from sklearn.model_selection import KFold
import numpy as np
import tensorflow as tf
import time
start_time = time.time()
# 1.47
current_Env = Env()
current_Env.readData()
current_Env.processing()
current_Env.appendSeq()



cross_validation = 4
percisionListPart = []
percisionListAll = []
kf = KFold(n_splits=cross_validation)
cvIndex = 1

for trainUserIdRange, testUserIdRange in kf.split(np.array(range(current_Env.numUser))):  #current_Env.numUser
    # trainUserIdRange/testUserIdRange both are lists of UserId
    # initial network for each cv
    RL = DeepQNetwork( current_Env.n_actions, 164,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=5000,
                  batch_size=320,
                  # output_graph=True
                  )
    step = 0
    # total train episode
    for episode in range(4):
        # for each episode train all the user onece
        episode_start_time = time.time()
        for currentUserId in trainUserIdRange + 1:    #current_Env.numUser
            currentSeqIndex = 0
            observation = current_Env.generateInputVector( currentUserId, currentSeqIndex )
            while True:
                # RL choose action based on observation which its a index number
                flag, actionIndex = RL.choose_action(observation)
                action = current_Env.actionTransform(actionIndex)
                # RL take action and get next observation and reward
                ifTerminal, currentSeqIndex, observation_, reward = current_Env.update(currentUserId, currentSeqIndex, action)
                # Experience replay
                RL.store_transition(observation, actionIndex, reward, observation_)

                if (step > 200) and (step % 5 == 0):
                    RL.learn()

                # swap observation
                observation = observation_
                step += 1
                # break while loop when end of this episode
                if ifTerminal:
                    print("User: ", currentUserId, " Done")
                    break
        print("CV: ", cvIndex, " episode: ", episode + 1, " Done", " time cost: ", time.time() - episode_start_time, "s")

    # test1flag, test1action = RL.choose_action(current_Env.generateInputVector( 1, 0 ))
    # print(test1flag, current_Env.allSeqList[test1action])

    # end of game
    print('Training is over now its testing')
    hitListPart = []
    hitListAll = []
    for currentUserId in testUserIdRange + 1:
        usersAllPreference = current_Env.getUserAllpreference(currentUserId)
        for currentSeqIndex in range(current_Env.getUserMaxSeqIndex(currentUserId)):
            flag, actionIndex = RL.choose_action( current_Env.generateInputVector( currentUserId, currentSeqIndex ) )
            hitCountPart = 0
            hitCountAll = 0
            for i in current_Env.allSeqList[actionIndex]:
                hitCountAll += 1 if i in usersAllPreference else 0
                hitListAll += [hitCountAll]
                hitCountPart += 1 if i in current_Env.generateRealActionNumber(currentUserId, currentSeqIndex) else 0
                hitListPart += [hitCountPart]

    precisionPart = sum( hitListPart ) / current_Env.numPredict / len( hitListPart )
    precisionAll = sum(hitListAll)/current_Env.numPredict/len(hitListAll)
    percisionListPart += [precisionPart]
    percisionListAll += [precisionAll]
    print( 'Testing is over' )
    print("for this cv, part scores: ", precisionPart, "all scores: ", precisionAll)
    if cvIndex == cross_validation:
        print("CV scores for part are: ",  percisionListPart, " the final profermence: ", sum(percisionListPart)/len(percisionListPart))
        print("CV scores for all are: ",  percisionListAll, " the final profermence: ", sum(percisionListAll)/len(percisionListAll))
        print( "--- %s seconds ---" % (time.time() - start_time) )
        # RL.plot_cost()
    cvIndex += 1



