import pandas as pd
import numpy as np
from scipy import spatial

class Env:
    def __init__(self):
        self.inputName = './ml-100k/'
        self.ratingPd = pd.DataFrame()
        self.userPd = pd.DataFrame()
        self.itemPd = pd.DataFrame()

        self.summaryPd = pd.DataFrame()
        self.userMovieSeq = pd.DataFrame()

        self.numUser = 0
        self.numUserHistory = 7
        self.numPredict = 10
        # action space get from training data
        self.allSeqList = []
        self.n_actions = 0

    # check
    def readData(self):
        # Reading users file:
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.userPd = pd.read_csv(self.inputName + "u.user", sep='|', names=u_cols, encoding='latin-1' )

        # Reading ratings file:
        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratingPd = pd.read_csv(self.inputName+ "u.data", sep='\t', names=r_cols, encoding='latin-1' )

        # Reading items file:
        i_cols = ['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
                  'Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.itemPd = pd.read_csv(self.inputName+ "u.item", sep='|', names=i_cols,
                             encoding='latin-1' )
        print("Data read")
    # check
    def processing(self):
        # processing for user info
        self.userPd = pd.get_dummies( self.userPd.drop( ["zip_code"], axis=1 ) )
        if_morethan_34 = np.where( self.userPd['age'] > 34, 1, 0 )
        self.userPd['if_morethan_34'] = if_morethan_34
        self.userPd = self.userPd.drop( ["age", "user_id"], axis=1 )

        # processing for rating info
        self.ratingPd = self.ratingPd.sort_values( ['user_id', 'unix_timestamp'] )
        self.ratingPd = self.ratingPd.drop( ["unix_timestamp"], axis=1 )
        self.ratingPd.index = range( self.ratingPd.shape[0] )

        # processing for item info
        self.itemPd = self.itemPd.drop(["release date", "video release date", "IMDb URL", "movie title", "movie_id"], axis = 1 )

        # # Getting the summary info
        # self.summaryPd = pd.merge( pd.merge( self.ratingPd, self.userPd, on="user_id" ), self.itemPd, how="left" )

        # Getting the Seq info
        self.userMovieSeq = self.ratingPd.groupby('user_id')['movie_id'].apply(list).reset_index(name='movieSeq')
        self.numUser = self.userMovieSeq.shape[0]
        print( "Data processed" )
    # check
    def generateInputVector(self, currentUserId, currentSeqIndex):    # assume all user had rated more than 6 movies
        if (currentSeqIndex > self.getUserMaxSeqIndex(currentUserId)):
            print("input index wrong",  end = " ")
            print("you are inputtng: ", currentSeqIndex, " for ", currentUserId)
            return
        UserVector = self.userPd.loc[currentUserId-1].values     # with out rating
        # print(UserVector)

        tripleMovie = self.userMovieSeq.loc[self.userMovieSeq['user_id'] == currentUserId, "movieSeq"].tolist()[0][currentSeqIndex:currentSeqIndex + self.numUserHistory]
        tripleMovieIndex = np.array(tripleMovie) - 1
        HistoryMovieVector = self.itemPd.loc[tripleMovieIndex].stack().values
        # print(HistoryMovieVector)

        tripleMoviePd = pd.DataFrame( tripleMovieIndex + 1, columns=["movie_id"] )
        # print(tripleMoviePd)
        tripRating = np.array( pd.merge( self.ratingPd.loc[self.ratingPd['user_id'] == currentUserId, ["movie_id","rating" ]], tripleMoviePd ).rating )
        # print(tripRating)
        # print(len(np.concatenate((UserVector, HistoryMovieVector, tripRating))))
        return np.concatenate((UserVector, HistoryMovieVector, tripRating))

    # if use users all preferences to calculate presicion for each prediction list
    def getUserAllpreference(self, currentUserId):
        return self.userMovieSeq.loc[self.userMovieSeq['user_id'] == currentUserId, "movieSeq"].tolist()[0]

    # if only use users next n preferences to calculate presicion for each prediction list
    def generateRealActionNumber(self, currentUserId, currentSeqIndex):
        tripleActionMovie = self.userMovieSeq.loc[self.userMovieSeq['user_id'] == currentUserId, "movieSeq"].tolist()[
                                0][currentSeqIndex + self.numUserHistory:currentSeqIndex + self.numUserHistory + self.numPredict]
        return tripleActionMovie
    def generateRealAction(self, currentUserId, currentSeqIndex):
        tripleActionMovie = self.generateRealActionNumber(currentUserId, currentSeqIndex)
        tripleActionMovieIndex = np.array( tripleActionMovie ) - 1
        ActionMovieVector = self.itemPd.loc[tripleActionMovieIndex].values
        return ActionMovieVector
    # check
    def getUserMaxSeqIndex(self, currentUserId):
        return  len(self.userMovieSeq.loc[self.userMovieSeq['user_id'] == currentUserId, "movieSeq"].tolist()[0]) - self.numUserHistory - self.numPredict

    # check
    def checkIfTerminal(self, currentUserId, currentSeqIndex):
        return (currentSeqIndex == self.getUserMaxSeqIndex(currentUserId) - 1)

    def computeSimilarity(self, array1, array2):
        return 1 - spatial.distance.cosine( array1, array2 )

    def computeOrderReward(self, order1, order2):
        # order1 order2 are the np.array in same d
        inter = order1 - order2
        re = np.dot( inter, inter )
        return 8 - re

    def computeReward(self, currentUserId, currentSeqIndex, predictAction):
        # @param currentUserId, currentSeqIndex to get the real ActionsVector
        # use cos similarity to compute reward1
        # use order to sompute reward2
        # assume all input is a array in 2D
        realAction = self.generateRealAction(currentUserId, currentSeqIndex)
        waitingList = list(range(len(predictAction)))    # index for realAction
        maxSimilarity = []
        predictorderList = []
        realOrderList = list(range(len(predictAction)))
        for i in range(len(predictAction)):       # index for predictAction
            similarity = [-1 for _ in range(self.numPredict)]
            for j in waitingList:
                similarity[j] = self.computeSimilarity(predictAction[i], realAction[j])
            maxSimilarity += [max(similarity)]
            maxIndex = similarity.index(max(similarity))
            predictorderList += [maxIndex]
            waitingList.remove(maxIndex)
        reward1 = sum(maxSimilarity)
        # print( "reward1:", reward1 )
        reward2 = self.computeOrderReward(np.array(predictorderList), np.array(realOrderList))
        # print( "reward2:", reward2/8 )
        # To be weighted
        return reward1*2 + reward2/8.0




        pass #return a int

    def update(self, currentUserId, currentSeqIndex, predictAction):
        ifTerminal = False
        reward = self.computeReward(currentUserId, currentSeqIndex, predictAction)
        if self.checkIfTerminal(currentUserId, currentSeqIndex):
            ifTerminal = True
        return (ifTerminal, currentSeqIndex + 1, self.generateInputVector(currentUserId, currentSeqIndex + 1), reward)
    #check

    def appendSeq(self):
        for i in range( self.userMovieSeq.shape[0]):
            l = self.userMovieSeq.iloc[i, 1]
            for i in range( len(l) - self.numPredict -1):
                self.allSeqList += [l[i:i + self.numPredict]]
        self.n_actions =  len(self.allSeqList)
        print("calculated there will be : ", self.n_actions, " actions")

    def actionTransform(self, actionIndex):
        tripleActionMovie = self.allSeqList[actionIndex]
        tripleActionMovieIndex = np.array( tripleActionMovie ) - 1
        ActionMovieVector = self.itemPd.loc[tripleActionMovieIndex].values
        return ActionMovieVector


# current_Env = Env()
# current_Env.readData()    # check
# current_Env.processing()  # check
# # numUser = current_Env.numUser   #check
# # print(numUser)
# # print(current_Env.generateRealAction(1, 0))   # 84-d
# # current_Env.computeReward(1,0, current_Env.generateRealAction(1, 0))
# current_Env.appendSeq()     #check
# current_Env.generateInputVector(1,0)



