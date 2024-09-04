import torch
import torch.nn as nn
import numpy as np
import sys, os
from torch.utils.data import Dataset, DataLoader, TensorDataset

##### SET UP CMB POWER SPECTRA RANGE #####

camb_ell_min          = 2
camb_ell_max          = 5000
camb_ell_range        = camb_ell_max - camb_ell_min

##### PICK DEVICE ON WHICH THE MODEL WILL BE TRAINED ON. WE RECOMMEND USING GPU FOR TRAINING #####
device                = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#CUDA for GPU
#CPU for CPU
#GPU is generally recommended for higher speed.

##### LOAD UP MEAN AND STD FOR INPUT AND OUTPUT #####
extrainfo=np.load("extra/extrainfo_hs_5mil.npy",allow_pickle=True)
X_mean=torch.Tensor(extrainfo.item()['X_mean'])#.to(device)
X_std=torch.Tensor(extrainfo.item()['X_std'])#.to(device)
Y_mean=torch.Tensor(extrainfo.item()['Y_mean']).to(device)
Y_std=torch.Tensor(extrainfo.item()['Y_std']).to(device)

##### LOAD UP COV MAT #####
covinv=np.load('/home/grads/data/yijie/mltrial/extra/cosvarinvTT.npy',allow_pickle=True)[:camb_ell_range,:camb_ell_range]
covinv=torch.Tensor(covinv).to(device) #This is inverse of the Covariance Matrix

##### START DEFINING THE MACHINE LEARNING MODULES #####

class Supact(nn.Module):
    # Dynamical activation function, returns:
    # h(x)=(gamma+(1+exp(-beta*x))^(-1)*(1-gamma))*x
    # gamma and beta are trainable parameters.
    # We chose the initial value for gamma to be all 1, and beta to be all 0
    def __init__(self, in_size):
        super(Supact, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(in_size))
        self.beta = nn.Parameter(torch.zeros(in_size))
        self.m = nn.Sigmoid()
    def forward(self, x):
        inv = self.m(torch.mul(self.beta,x))
        fac = 1-self.gamma
        mult = self.gamma + torch.mul(inv,fac)
        return torch.mul(mult,x)

class Affine(nn.Module):
    def __init__(self):
        # This function is designed for the Neuro-network to learn how to normalize the data between
        # layers. we will initiate gains and bias both at 1 
        
        super(Affine, self).__init__()

        
        self.gain = nn.Parameter(torch.ones(1))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        return x * self.gain + self.bias

class Better_Attention(nn.Module):
    # This is the Dot-Product Attention Layer. Returns:
    # Attention(Q,K,V)=Softmax(QK^T/sqrt{d_K})V
    def __init__(self, in_size ,n_partitions):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        out = torch.reshape(prod,(batch_size,-1))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    # This is the Transformer Block. The input data vectors are splitted into channels and then pass through
    # DIFFERENT MLP Blocks(Note that this is different from the original transformer design).
    # Adding residual at the end before output.
    def __init__(self, in_size, n_partitions):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = Supact(in_size)#nn.Tanh()#nn.ReLU()#
        self.norm         = Affine()

        # set up weight matrices and bias vectors
        weights = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights = nn.Parameter(weights) # turn the weights tensor into trainable weights
        bias = torch.Tensor(in_size)
        self.bias = nn.Parameter(bias) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5)) # matrix weights init 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights) # fan_in in the input size, fan out is the output size but it is not use here
        bound = 1 / np.sqrt(fan_in) 
        nn.init.uniform_(self.bias, -bound, bound) # bias weights init

    def forward(self,x):
        mat = torch.block_diag(*self.weights) 
        x_norm = self.norm(x)
        o = self.act(torch.matmul(x_norm,mat)+self.bias)
        return o+x

class ResBlock(nn.Module):
    #Residual Dense Block. The input goes through two dense layers and then add input to output(Residual part).
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        #get/set up parameters
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = Supact(in_size)#nn.Tanh()#nn.ReLU()#
        self.act2 = Supact(in_size)#nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.layer1(self.norm1(x)))
        o2 = self.act2(self.layer2(self.norm2(o1))) + xskip

        return o2


class ResMLP(nn.Module):
    # ResMLP architecture:
    # Input Layer
    # N_block amount of ResBlocks
    # Output Layer

    def __init__(self, input_dim, output_dim, int_dim, N_block):

        super(ResMLP, self).__init__()

        modules=[]

        # Def: we will set the internal dimension as multiple of 128 (reason: just simplicity)
        int_dim = int_dim * 128

        # Def: we will only change the dimension of the datavector using linear transformations  
        modules.append(nn.Linear(input_dim, int_dim))
        
        # Def: by design, a pure block has the input and output dimension to be the same
        for n in range(N_block):
            # One full ResBlock contains 1 ResLayer + 1 Activation Function 
            modules.append(ResBlock(int_dim, int_dim))
            modules.append(Supact(int_dim))
        
        # Def: the transformation from the internal dimension to the output dimension of the
        #      data vector we intend to emulate
        
        modules.append(nn.Linear(int_dim, output_dim))
        modules.append(Affine())
        # NN.SEQUENTIAL is a PYTHORCH function DEFINED AT: https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # This function stacks up layers in the modules-list in sequence to create the whole model
        self.resmlp =nn.Sequential(*modules)

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.resmlp(x)

        return out




class TRF(nn.Module):
    # Transformer architecture: 
    # Input Layer
    # 3 ResBlocks
    # 1 Attention Layer+Transformer Block
    # output layer
    def __init__(self, input_dim, output_dim, int_dim, int_trf, N_channels):

        super(TRF, self).__init__()

        modules=[]

        int_dim = int_dim * 128
        
        n_channels = N_channels
        int_dim_trf = int_trf
        modules.append(nn.Linear(input_dim, int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(Supact(int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(Supact(int_dim))
        modules.append(ResBlock(int_dim, int_dim))
        modules.append(Supact(int_dim))
        modules.append(nn.Linear(int_dim, int_dim_trf))
        modules.append(Supact(int_dim_trf))
        modules.append(Better_Attention(int_dim_trf, n_channels))
        modules.append(Better_Transformer(int_dim_trf, n_channels))
        modules.append(nn.Linear(int_dim_trf, output_dim))
        modules.append(Affine())

        self.trf =nn.Sequential(*modules)
        

    def forward(self, x):
        #x is a cosmological parameter set you feed in the model
        out = self.trf(x)
        
        return out

#load in data
train_samples=np.load('parametersamples/hypersphere/coshs_train.npy',allow_pickle=True)

input_size=len(train_samples[0])

validation_samples=np.load('parametersamples/hypersphere/coshs_vali.npy',allow_pickle=True)

train_data_vectors=np.load('/home/grads/yijie/coshs_TT_train.npy',allow_pickle=True,mmap_mode='r+')

validation_data_vectors=np.load('datavectors/hypersphere/coshs_TT_vali.npy',allow_pickle=True)

out_size=len(train_data_vectors[0])

#assign training and validation sets

train_samples=torch.Tensor(train_samples)
train_data_vectors=torch.Tensor(train_data_vectors)
validation_samples=torch.Tensor(validation_samples)
validation_data_vectors=torch.Tensor(validation_data_vectors)



#normalizing samples and to mean 0, std 1

X_train=(train_samples-X_mean)/X_std
X_train[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization

X_validation=(validation_samples-X_mean)/X_std
X_validation[:,6:]=0 # we didn't vary the last 3 parameters: mnu, w, and wa in this test, so setting them to 0 automatically after normalization

X_train=X_train.to(torch.float32)
X_validation=X_validation.to(torch.float32)

X_mean=X_mean.to(device)
X_std=X_std.to(device)
#load the data to batches. Do not send those to device yet to save space


trainset    = TensorDataset(X_train, train_data_vectors)
validset    = TensorDataset(X_validation,validation_data_vectors)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)



#training
batch_size = 512
n_epoch = 900
cset = [16,32]#set of channels that we will test on

intdim = 4
int_trf = 5120
for nc in cset:
    #Set up the model and optimizer
    model = TRF(input_dim=input_size,output_dim=out_size,int_dim=intdim, int_trf=int_trf,N_channels=nc)
    

    model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0)


    losses_train = []
    losses_vali = []
    losses_train_med = []
    losses_vali_med = []



    reduce_lr = True#reducing learning rate on plateau
    if reduce_lr==True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=15)

    for n in range(n_epoch):
        losses=[]
        for i, data in enumerate(trainloader):
            model.train()

            X = data[0].to(device)# send to device one by one
            
            Y_batch = data[1].to(device)# send to device one by one
            Y_pred  = model(X).to(device)# model output is normalized C_ell*e^(2tau_re)/As
            #scale the output back to C_ell unit
            As = torch.exp(X[:,5]*X_std[0,5]+X_mean[0,5])
            exptau = torch.exp(2*X[:,3]*X_std[0,3]+2*X_mean[0,3])
            Y_pred =  Y_pred*Y_std+Y_mean
            Y_pred = Y_pred *As[:,None]/exptau[:,None]
            diff = Y_pred - Y_batch

        
            loss1 = torch.diag(diff @ covinv @ torch.t(diff))
            loss1 = torch.sqrt(1+2*loss1)
            loss=torch.mean(loss1)
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses_train.append(np.mean(losses))# We take means since a loss function should return a single real number
        losses_train_med.append(np.median(losses))

        with torch.no_grad():
            model.eval()
        
            losses = []
            for i, data in enumerate(validloader):
                X_v       = data[0].to(device)
                
                Y_v_batch = data[1].to(device)
                Y_v_pred = model(X_v).to(device)
                As=torch.exp(X_v[:,5]*X_std[0,5]+X_mean[0,5])
                exptau=torch.exp(2*X_v[:,3]*X_std[0,3]+2*X_mean[0,3])
                #Y_v_pred = Y_v_pred*Y_std2+Y_mean2
                #Y_v_pred = torch.matmul(Y_v_pred, transform_matrix)
                Y_v_pred_back = Y_v_pred*Y_std+Y_mean
                Y_v_pred_back = Y_v_pred_back*As[:,None]/exptau[:,None]
                v_diff = (Y_v_batch - Y_v_pred_back)
                
                
                loss1 = torch.diag(v_diff @ covinv @ torch.t(v_diff))# implement with torch.einsum
                loss1 = torch.sqrt(1+2*loss1)
                loss_vali=torch.mean(loss1)
                losses.append(loss_vali.cpu().detach().numpy())

            losses_vali.append(np.mean(losses))
            losses_vali_med.append(np.median(losses))

            if reduce_lr == True:
                print('Reduce LR on plateu: ',reduce_lr)
                scheduler.step(losses_vali[n])

        print('epoch {}, loss={}, validation loss={}, lr={}, wd={})'.format(
                        n,
                        losses_train[-1],
                        losses_vali[-1],
                        optimizer.param_groups[0]['lr'],
                        optimizer.param_groups[0]['weight_decay']
                        
                    ))



    PATH = "./trainedemu5000hstrf/chiTTAstautrfhshalfmil1trf5120evanc"+str(nc)
    torch.save(model.state_dict(), PATH+'.pt')
    extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std,'Y_mean2':Y_mean2,'Y_std2':Y_std2}
    np.save(PATH+'.npy',extrainfo)
    np.save(PATH+'losstrain.npy',losses_train)
    np.save(PATH+'lossvali.npy',losses_vali)
    np.save(PATH+'losstrainmed.npy',losses_train_med)
    np.save(PATH+'lossvalimed.npy',losses_vali_med)
