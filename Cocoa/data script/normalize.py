import numpy as np
from sklearn.decomposition import IncrementalPCA

'''
This file will generate the mean and std of the data set after
rescaled in to C'_ell=C_ll/As*exp(tau).
Optionally, the users can also perform PCA on the rescaled and
normalized data set, and then compute the mean and std of 
the PCA components as well.
'''

camb_ell_min          = 2#30
camb_ell_max          = 5000
camb_ell_range        = camb_ell_max  - camb_ell_min 
# Load the data
train_samples=np.load('YZ_samples/Planck256/inputnew/cos_pkc_T256_fifthtest.npy',allow_pickle=True)

train_data_vectors=np.load('YZ_samples/Planck256/lensed/cos_pkc_T256_fifthtest_TT.npy',allow_pickle=True)

#Rescale the data. My parameters are:[Omega_bh^2, Omega_ch^2, H0, tau_re ns, log(As10^10), mnu=0.06, w=-1, wa=0]
for i in range(len(train_data_vectors)):
    train_data_vectors[i]=train_data_vectors[i]/(np.exp(train_samples[i,5]))*(np.exp(2*train_samples[i,3]))

#Compute the mean and std of input and rescaled output
X_mean=train_samples.mean(axis=0, keepdims=True)
X_std=train_samples.std(axis=0, keepdims=True)
Y_mean=train_data_vectors.mean(axis=0, keepdims=True)
Y_std=train_data_vectors.std(axis=0, keepdims=True)

#PCA the rescaled normalized output
X=(train_data_vectors-Y_mean)/Y_std
n_pca=96
batchsize=43
PCA = IncrementalPCA(n_components=n_pca,batch_size=batchsize)

for batch in np.array_split(X, batchsize):
    PCA.partial_fit(batch)

train_pca=PCA.transform(X)

#Compute the mean and std of PCA output
Y_mean2 = train_pca.mean(axis=0, keepdims=True)
Y_std2 = train_pca.std(axis=0, keepdims=True)

#save the info for training
np.save('YZ_samples/PCAcomp/PCAmat_T256_tt.npy',PCA.components_)#PCA matrix
extrainfo={'X_mean':X_mean,'X_std':X_std,'Y_mean':Y_mean,'Y_std':Y_std,'Y_mean2':Y_mean2,'Y_std2':Y_std2}
np.save('YZ_samples/PCAcomp/extrainfo_plk_tt_T256.npy',extrainfo)
