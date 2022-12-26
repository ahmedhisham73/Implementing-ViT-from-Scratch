#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# In[21]:


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.models import Model


# In[22]:


'''
define function of the vision transformer 
it takes the configuration file that's present in main 
step 1 
inputs of ViT 
-input layer with proper input shape 
nb: input shape = number_of_patches*Patch_size(w,h)*RGB
-number_of_patches is taken from the configuration created in the main 
-Patch_size is taken from the configuration created in main
-input shape should equal N*ph*pw*3 
N= Hxw/ph*pw= 512*512/32*32 =256

-input shape -(256,32*32*3)-> (none,256,3072)



step2 patch+ position embeddings
-patch+postion_embedding -> linearly embedded feeded will give a resulting sequence vector -> Dense Layer
-the resulting sequence will be fed into the transformer encoder 
-input of the patchand embedding will take the number of hidden layers

-position_embedding : range function starts from zero positional encoding and ends at the number of patches
with delta paramter which is the number of incremental iterations 
postions are a tensor from 0 to 255
shape of the position is 256 


step3 combine the patch embedding + positions -> position embedding
-position embedding -> Embedding Layer
Embedding_layer
*its input is coming from the position
*input dimension
*output dimension
since the output is needed to fed into the transformer 



step4 add the classToken

build a separate class before the ViT and name it ClassToken
-the class token is a layer
-inside it define build function which contains weight initialization
-call_extractor function


Step5 finalize the input for the transformer 
-concatenate the token and embedding


step6 start building the transformer encoder

from the paper the transformer encoder takes the input data -> concatenated embeddings and the configuartions
transformer encoder starts by a skip connection through normalization layer followed by MultiHeadAttention


in MultiHead attention layer 
1-number of heads =number of heads in the configuration file 
2-key dimensions = hidden layers 



declare mlp function 


go to ViT and repeat the transformer = 12 = number of layers

add the classification task 

'''

class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call_feature(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls
    

    
    
    
    
    
    
    
    
    
    
    
    



    
    
    
    


# In[30]:


def MultiLayerPreceptron(x,cf):
    x = Dense(cf["mlp_dim"], activation="gelu")(x)
    x = Dropout(cf["drop_out_rate"])(x)
    x = Dense(cf["hidden_size"])(x)
    x = Dropout(cf["drop_out_rate"])(x)
    return x


# In[31]:


def transformer_encoder(x,cf):
    skip_connection_1=x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(
    num_heads= cf["number_of_heads"],
        key_dim= cf["hidden_size"])(x,x)
    
    x = Add()([x, skip_connection_1])
    
    
    
    skip_connection_2 = x
    x = LayerNormalization()(x)
    x = MultiLayerPreceptron(x, cf)
    x = Add()([x, skip_connection_2])
    
    return x 
    
    


# In[32]:


def ViT(cf):
    
    #input pipeline for the Model
    
    input_shape=(cf["num_patches"],cf["patch_size"]*cf["patch_size"]*cf["number_of_channels"])
    inputs=Input(input_shape)
    patch_embedding = Dense(cf["hidden_size"])(inputs)
    positions= tf.range(start=0,limit=cf["num_patches"],delta=1)
    
    
    
    position_embedding=Embedding(input_dim=cf["num_patches"],output_dim=cf["hidden_size"])(positions)
    
    embedd=patch_embedding+position_embedding
    token = ClassToken()(embedd)
    x = Concatenate(axis=1)([token, embedd]) 
    
    #transformer itself
    for _ in range(cf["num_layers"]):
    
        x = transformer_encoder(x, cf)
        
        
        
    x = LayerNormalization()(x) ## (None, 257, 768)
    x = x[:, 0, :] ## (None, 768)
    x = Dropout(0.1)(x)
    x = Dense(10, activation="softmax")(x)

    model = Model(inputs, x)
    return model
        
        
    
    
    
    


# In[33]:


'''
image size = wxhxc -> image width x imageheight x number of channels 
patch size = pw x ph

paramters of ViT
1-number oflayers
2- hidden size 
3-multi layer preceptron dimensions 
4-number of heads 
5-Drop out rate 
6-patch size
7-number of patches = image_size^2/patch_size^2
8-number of channels 

'''

if __name__ == "__main__":
    config={}
    config["num_layers"]=12
    config["hidden_size"]=768
    config["mlp_dim"]=3072
    config["number_of_heads"]=12
    config["drop_out_rate"]=0.1
    config["num_patches"] = 256
    config["patch_size"]=32
    config["number_of_channels"]=3
    model = ViT(config)
    model.summary()


# In[ ]:




