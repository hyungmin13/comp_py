#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import interpn
import h5py
from scipy.io import loadmat
import argparse
from Tecplot_mesh import tecplot_Mesh
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, grids
    
def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_x, out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def Derivatives(dynamic_params, all_params, g_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']

    all_params["network"]["layers"] = dynamic_params
    out, out_t = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)    

    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]
    p = 1.185*all_params["data"]['u_ref']*out[:,3:4]
    uvwp = np.concatenate([u, v, w, p],1)
    ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["data"]["domain_range"]["t"][1]
    vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["data"]["domain_range"]["t"][1]
    wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["data"]["domain_range"]["t"][1]

    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["data"]["domain_range"]["x"][1]
    vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["data"]["domain_range"]["x"][1]
    wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["data"]["domain_range"]["x"][1]

    uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["data"]["domain_range"]["y"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["data"]["domain_range"]["y"][1]
    wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["data"]["domain_range"]["y"][1]

    uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["data"]["domain_range"]["z"][1]
    vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["data"]["domain_range"]["z"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["data"]["domain_range"]["z"][1]
    
    
    acc_x = ut + u*ux + v*uy + w*uz
    acc_y = vt + u*vx + v*vy + w*vz
    acc_z = wt + u*wx + v*wy + w*wz
    acc = np.concatenate([acc_x, acc_y, acc_z],1)

    matrix1 = [ux, uy, uz, vx, vy, vz, wx, wy, wz]
    matrix2 = [ux, vx, wx, uy, vy, wy, uz, vz, wz]
    vor_mag = np.sqrt((uz-wx)**2+(uy-vx)**2+(vz-wy)**2)
    Q = 0
    for i,j in zip(matrix1, matrix2):
        S = 0.5*(i + j)
        P = 0.5*(i - j)
        Q = Q + 0.5* ((np.abs(P))**2 - (np.abs(S))**2)

    return uvwp, acc, vor_mag, Q

#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(description='RBC_PINN')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    #parser.add_argumnet('-t', '--timestep', type=int, help='timestep', default="")
    args = parser.parse_args()
    checkpoint_fol = args.checkpoint

    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['domain_init_kwargs']['grid_size'] = [101, 120, 120, 120]
    #a['data_init_kwargs']['path'] = '/scratch/hyun/UrbanRescue/run065/'
    #a['problem_init_kwargs']['path_s'] = 'Ground/'
    #with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
    #    pickle.dump(a,f)

    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[4].split('.')[0]))
    with open(checkpoint_list[-1],'rb') as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, grids = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params

#%%
    pos_ref = all_params["domain"]["in_max"].flatten()
    vel_ref = np.array([all_params["data"]["u_ref"],
                        all_params["data"]["v_ref"],
                        all_params["data"]["w_ref"]])

    ref_key = ['t_ref', 'x_ref', 'y_ref', 'z_ref', 'u_ref', 'v_ref', 'w_ref']
    ref_data = {ref_key[i]:ref_val for i, ref_val in enumerate(np.concatenate([pos_ref,vel_ref]))}

#%%
    for s in range(5):
        timestep =s+47
        mesh_xyz = np.meshgrid(grids['eqns']['x'], grids['eqns']['y'], grids['eqns']['z'], indexing='ij')
        shape = mesh_xyz[0].reshape(-1).shape[0]
        eval_grid = np.concatenate([np.zeros((shape,1))+grids['eqns']['t'][timestep],mesh_xyz[0].reshape(-1,1),
                                    mesh_xyz[1].reshape(-1,1),mesh_xyz[2].reshape(-1,1)],1)
        x_e = eval_grid[:,1].reshape(120,120,120)*ref_data['x_ref']
        y_e = eval_grid[:,2].reshape(120,120,120)*ref_data['y_ref']
        z_e = eval_grid[:,3].reshape(120,120,120)*ref_data['z_ref']

#%%
        dynamic_params = all_params["network"].pop("layers")
        uvwp, acc, vor_mag, Q = zip(*[Derivatives(dynamic_params, all_params, eval_grid[i:i+10000], model_fn) 
                                for i in range(0, eval_grid.shape[0], 10000)])
        uvwp = np.concatenate(uvwp, axis=0)
        vor_mag = np.concatenate(vor_mag, axis=0)
        acc = np.concatenate(acc, axis=0)
        Q = np.concatenate(Q, axis=0)

        filename = "Tecplot_data/"+checkpoint_fol+"/QUD_eval_"+str(timestep)+".dat"
        if os.path.isdir("Tecplot_data/"+checkpoint_fol):
            pass
        else:
            os.mkdir("Tecplot_data/"+checkpoint_fol)
        X, Y, Z = (x_e[0,0,:].shape[0], y_e[0,:,0].shape[0], z_e[:,0,0].shape[0])
        vars = [('u_pred[m/s]',np.float32(uvwp[:,0].reshape(-1))), ('v_pred[m/s]',uvwp[:,1].reshape(-1)),
                ('w_pred[m/s]',uvwp[:,2].reshape(-1)), ('p_pred[Pa]',uvwp[:,3].reshape(-1)),
                ('acc_x [m/s^2]',acc[:,0].reshape(-1)), ('acc_y [m/s^2]',acc[:,1].reshape(-1)),
                ('acc_z [m/s^2]',acc[:,2].reshape(-1)),
                ('vormag[1/s]',vor_mag.reshape(-1)), ('Q[1/s^2]', Q.reshape(-1))]
        fw = 27
        tecplot_Mesh(filename, X, Y, Z, x_e.reshape(-1), y_e.reshape(-1), z_e.reshape(-1), vars, fw)
        total = np.concatenate([uvwp, acc, vor_mag.reshape(-1,1), Q.reshape(-1,1)],1)
        with open("Tecplot_data/"+checkpoint_fol+"/QUD_eval_"+str(timestep)+".pickle", 'wb') as f:
            pickle.dump(total, f)

