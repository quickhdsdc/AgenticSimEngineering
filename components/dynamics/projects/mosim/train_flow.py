from MoSim.src.tools import initialize_residual_flow, train_residual_flow

device = 'cuda:2'
ckpt_path = '/projects/Neural-Simulator/data/archive/cheetah_old/cheetah_random_test/'
K = 16
latent_size = 24
hidden_units = 128
hidden_layers = 3
state_dim = 24
batch_size = 50000
if_ckpt = '/projects/Neural-Simulator/ckpt/flow/cheetah/cheetah_02111639/ckpt_1206M.pth'
nfm = initialize_residual_flow(K,latent_size,hidden_units,hidden_layers,state_dim, device, ckpt_path, batch_size, if_ckpt)


batch_size = 50000
data_path = '/Neural-Simulator/data/archive/cheetah_old/cheetah_random/'
data_path_test = '/projects/Neural-Simulator/data/archive/cheetah_old/cheetah_random_test/'
save_path  = '/projects/Neural-Simulator/ckpt/flow/cheetah'
task_name = 'cheetah'
lr = 3e-5
train_residual_flow(nfm,batch_size,data_path,data_path_test,device,save_path, task_name, lr)

