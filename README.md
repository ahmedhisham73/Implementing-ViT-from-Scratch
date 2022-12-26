# Implementing-ViT-from-Scratch

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
