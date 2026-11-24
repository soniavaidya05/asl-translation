"""
1. Start with the encoder which recieves the results of the channel specific transformer
   We use the results of the channel specific encoder to feed the multichannel encoder
"""
import math
import torch
# This stores parameters to be learned automatically
import torch.nn as nn
# This is used for functions that do not have learned parameters
import torch.nn.functional as F


# Define the Multi-channel encoder via a class
# The input to this would be a batch of videos, each with T frames, each frame has three channels of 256 dimensions

def make_ff(d, d_ff):
    """
    Creates a neural network 
    First Layer: takes a input of d to d_ff hidden units
    Second: RELY activation function
    Final layer: takes a hidden unit to ouput of dim d
    """
    return nn.Sequential(nn.Linear(d, d_ff), nn.RELU(), nn.Linear(d_ff, d))

def attend(query, Ks, Vs):

    # compute the dot product of query with all the ks
    dot_product = Ks @ query

    # apply the softmax()
    dot_product = F.softmax(dot_product / math.sqrt(d))

    # apply to the key values
    return Vs @ dot_product

class MultiChannelAttention(nn.Module):
    # we inherit from the nn.Module so that we inherit its attributes required for Pytorch training
    # essentially it becomes a Pytorch class
    def __init__(self, num_channels, embedding_size, hidden_units):
        super.__init__()
        self.num_channels = num_channels
        def mk(f):
            # creates a list of modules given a function 
            # we need it to ensure that we apply the same function for the three articulators
            # f is a module like linear matrix, layer norm, RELU etc.
            return nn.ModuleList([f() for i in range(num_channels)])
        # now we initialize the attributes that "hold" the different parts of the multi-channel architecture
        # Note, that we are pass a function that returns our desired function because 
        # mk() calls each f() with no input so we're calling the wrapper function that returns the input
        # First: normalization layers 
        # Norm applied before attention block
        self.norm1 = mk(lambda: nn.LayerNorm(embedding_size))
        # Norm applied before the FFN
        self.norm2 = mk(lambda: nn.LayerNorm(embedding_size))
        # QKV matrices
        self.q = mk(lambda: nn.Linear(embedding_size,embedding_size))
        self.k = mk(lambda: nn.Linear(embedding_size,embedding_size))
        self.v = mk(lambda: nn.Linear(embedding_size,embedding_size))
        # FFN

        self.ffn = mk(lambda: make_ff(embedding_size, hidden_units))

    def forward(self, input, pad_masks):
        """
        This actually computes the result for each channel in one input
        input = [hand_embeddings_across_frames, mouth_embeddings_across_frames, pose_embeddings_across_frames]
        input has passed through the channel specific section
        The pad masks, these are computed somewhere else, and they essentially go through all the videos and checks the maximum length, then loops
        through all the vidoes and checks if it < max len. Anything that is < will be padded i.e the padmasks is set to true for the frames that are padded
        In the same way as input it is input_masks = [hand_mask, mouth_mask, pose_mask]

        Why is this important: because padded "embeddings" should not contribute to the attention block
        """
        l = []
        # We do the normalization first because in Fig 2 they apply layer norm before doing the QKV calculation
        # We have seperate for l ("Q"), K and V
        for i in range(self.num_channels):
            # per channel we will do the normalization and note down the results
            norm_func = self.norm1[i]
            l.append(norm_func(input[i]))
        K = []
        for i in range(self.num_channels):
            # per channel we will multiiply with K and note down the results
            k_res = self.k[i]
            K.append(k_res(l[i]))
        V = []
        for i in range(self.num_channels):
            # per channel we will multiiply with V and note down the results
            q_res = self.v[i]
            V.append(k_res(l[i]))
        # So for each channel we have established the keys and values for each channel
        # These will be continually referenced during the attention blocks
        
        for i in range(self.num_channels):
            # set the query which is the current channel -> make sure you use the normalized version 
            q_res = self.q[i](l[i])

            # collect the keys and values we will attend to 
            possible_K = []
            possible_V = []
            mask = []
            for j in range(self.num_channels):
                if i != j:
                    possible_K.append(K[j])
                    possible_V.append(V[j])
                    mask.append(pad_masks[j])
            # stack vertically, i.e along the time axis
            possible_K = torch.cat(possible_K, dim=1)
            possible_V = torch.cat(possible_V, dim=1)
            mask = torch.cat(mask, dim=1)
            # TODO: fix this function
            weighted_vector = attend(q_res, possible_K, possible_V)
            # the weighted vector for this channel is added to the norm value, because of residual connections
            # we dont replace the vector but add to it so that its not learning a new vector all over again
            x = input[i] 
            x += weighted_vector
            # now we normalize 
            norm2_func = self.norm2[i]
            x = norm2_func(x)
            # perform ffn 

            ffn_func = self.ffn[i]
            ffn_x = ffn_func(x)
            x += ffn_x
            