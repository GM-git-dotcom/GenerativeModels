import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header = None, engine = 'python', 
                     encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header = None, engine = 'python', 
                     encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header = None, engine = 'python', 
                     encoding = 'latin-1')
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int64')
#Now, we will make 2 matrices : one for the training set and the other for test set
#In these, the rows will be the users, columns will be the movies, and
#cells will contain the ratings given. If a particular user didn't rate a movie,
#that cell will contain 0. Both sets will have same (total from dataset) number of
#lines and columns

#To get the max users, we get the max of the maxima of test and training set users
#This is because the users may be randomly arranged in both sets

nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

#Converting the data into an array with users in lines and movies in columns
#Features are columns, observations are rows
def convert(data):
    #We will be making a list of lists (not 2D numpy array for now)
    #It will contain 943 lists (as there are 943 users)
    #Inner contents will be the 1682 ratings for movies by index, 0 if not rated
    new_data = [] #final list that we will return
    for id_users in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_users] #contains the IDs of the movies rated by the id_user
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings)) #using list() just to make sure it is a list!
    return (new_data)
training_set = convert(training_set)
test_set = convert(test_set)

#Converting the data into Torch tensors
#Tensors are multidimensional matrices having the same datatype as elements
#We will be working with PyTorch tensors. There even exist TensorFlow tensors for TF.

training_set = torch.FloatTensor(training_set) #FloatTensor class expects a list of lists!
test_set = torch.FloatTensor(test_set)
#Executing above 2 lines will make training_set and test_set disappear from variable explorer
#as Spyder does not recognize Torch tensors yet.

#Converting the ratings into binary ratings 1 (linked) or 0 (not linked)
training_set[training_set == 0] = -1 #Converting all 0s to -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

#Creating the architecture of the NN
#Probabilistic Graphical Model is how RBM works
class RBM:
    def __init__(self, nv, nh):
        self.W=torch.randn(nh, nv) #random weights 
        #Initializing weights, in a Torch tensor -- these weights are probabilities of
        #the visible nodes given the hidden nodes (v|h). randn(nh, nv) initializes weights of 
        #size (nh, nv) as a normal distribution with mean 0 and variance 1
        self.a = torch.randn(1, nh) #Bias for hidden nodes, giving 2 dimensions as PyTorch cannot
        #take single dimension, it always wants 2D so value 1 is the batch_size for sampling 
        #from the distribution
        self.b = torch.randn(1, nv) #Bias for visible nodes
    #Now the 2nd function is about sampling the hidden node according to the probability
    #p(h|v) which is nothing else but the sigmoid activation function
    def sample_h(self, x): #x is the matrix values of visible nodes v in the probability p(h|v)
        wx = torch.mm(x, self.W.t()) #Transpose!
        activation = wx + self.a.expand_as(wx) #Making sure the bias is applied to EACH line of the minibatch
        #This activation function represents a probability that the hidden node will be activated
        #according to the value of the visible node
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
        #p_h_given_v vector has, for example if there are 100 hidden nodes, it has 100 probabilities
        #whether or not those nodes will be activated given the values of v
        #Now we use these probabilities to sample the activation of each of these 100 hidden nodes
        #Sampling - we take a random number between 0 to 1, if this number is larger than ith node
        #probability, then the node doesn't activate and vice versa (Bernoulli Sampling)
        #In the end, we get a vector of 0s and 1s stating which nodes got activated and which didn't.
    def sample_v(self, y): #values of the visible node is y
        wy = torch.mm(y, self.W) #not taking transpose as it is p_v_given_h
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    #The last function is about Contrastive Divergence that we will use to approximate the log 
    #likelyhood gradient. Remember RBM is an Energy Based Model so our goal is to optimize weights
    #such that the energy is minimized. Similarly, it can also be said that RBM is a probabilistic
    #graphical model where the goal is to maximize the log likelyhood (equivalent to minimizing E)
    #Log likelyhood gradient -- We have to find the gradient which is very computationally expensive.
    #So we approximate the gradient via Gibbs' Sampling. It consists of creating a Gibbs' Chain
    #in k steps by sampling the visible and hidden nodes k times (k iterations):
    #v0 --> h0 --> v1 --> h1 --> ... We create the algorithm for this in train function below.
    def train(self, v0, vk, ph0, phk): #v0: initial visible node values, vk: values of visible nodes
    #after k steps of k CDs, ph0: probability vector that at first iteration the hidden nodes equal 
    #1 given the visible nodes, phk: probabilities of hidden nodes after k sampling given vk
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0) #Summing with 0 to keep the format of b as tensor of 2D
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0]) #Number of features or movies. Can also take nb_movies
nh = 100 #This we can tune. Here we detect 100 features
batch_size = 100 #tunable
rbm = RBM(nv, nh) #model created

#Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    #Now we create a loss function
    train_loss = 0
    #To normalize this train loss, we use a counter 
    s = 0.0 #will increment after each epoch
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10): #for the k steps of CD
            _, hk = rbm.sample_h(vk) #Putting vk not v0 so v0 doesn't change as it is TARGET
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] #Freezing the -1 values so they don't get trained
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[vk >= 0]))
        s += 1.0
    print("Epoch: "+ str(epoch) + " Loss: " + str(train_loss/s))
    
#Testing the RBM
#Now we create a loss function
test_loss = 0
#To normalize this train loss, we use a counter 
s = 0.0 #will increment after each epoch
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0: 
        _, h = rbm.sample_h(v) #Putting vk not v0 so v0 doesn't change as it is TARGET
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.0
print("Test loss: "+str (test_loss/s))
            
