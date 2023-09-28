import os
import sys
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()

case = "test/up"
case = "0.001/16"
CFD = os.path.abspath(f"data/{case}/data")

# project libraries
sys.path.append(os.path.join(CFD,"../PySPOD/"))

# Import library specific modules
from pyspod.spod.standard  import Standard  as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils     as utils_spod
import pyspod.utils.weights  as utils_weights
import pyspod.utils.errors   as utils_errors
import pyspod.utils.io       as utils_io
import pyspod.utils.postproc as post



## -------------------------------------------------------------------
## initialize MPI
## -------------------------------------------------------------------
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
except:
    comm = None
    rank = 0
## -------------------------------------------------------------------


## -------------------------------------------------------------------
## read data and params
## -------------------------------------------------------------------
## data
data_file = os.path.join(CFD, 'body_p.npy')
data = np.load(data_file).T
data.shape
dt = 0.005
nt = data.shape[0]
x1 = np.linspace(-0.35, 0.35, data.shape[1])
x2 = np.linspace(-0.35, 0.35, data.shape[2])
## params
config_file = os.path.abspath(os.path.join(CFD, '../../../../spod/input_spod.yaml'))
params = utils_io.read_config(config_file)
params['time_step'] = dt
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## compute spod modes and check orthogonality
## -------------------------------------------------------------------
standard  = spod_standard (params=params, comm=comm)
# streaming = spod_streaming(params=params, comm=comm)
spod = standard.fit(data_list=data)
results_dir = spod.savedir_sim
# flag, ortho = utils_spod.check_orthogonality(
#     results_dir=results_dir, mode_idx1=[1],
#     mode_idx2=[0], freq_idx=[5], dtype='double',
#     comm=comm)
# print(f'flag = {flag},  ortho = {ortho}')
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## compute coefficients
## -------------------------------------------------------------------
# file_coeffs, coeffs_dir = utils_spod.compute_coeffs_op(
#     data=data, results_dir=results_dir, comm=comm)
## -------------------------------------------------------------------



## -------------------------------------------------------------------
## compute reconstruction
## -------------------------------------------------------------------
# file_dynamics, coeffs_dir = utils_spod.compute_reconstruction(
#     coeffs_dir=coeffs_dir, time_idx='all', comm=comm)
## -------------------------------------------------------------------



## only rank 0
if rank == 0:
    ## ---------------------------------------------------------------
    ## postprocessing
    ## ---------------------------------------------------------------
    ## plot eigenvalues
    spod.plot_eigs(filename='eigs.png')
    spod.plot_eigs_vs_frequency(filename='eigs_freq.png')
    spod.plot_eigs_vs_period(filename='eigs_period.png')

    ## identify frequency of interest
    T1 = 0.9; T2 = 4
    f1, f1_idx = spod.find_nearest_freq(freq_req=2, freq=spod.freq)
    f2, f2_idx = spod.find_nearest_freq(freq_req=4, freq=spod.freq)
    f3, f3_idx = spod.find_nearest_freq(freq_req=6, freq=spod.freq)
    f4, f4_idx = spod.find_nearest_freq(freq_req=8, freq=spod.freq)
    f5, f5_idx = spod.find_nearest_freq(freq_req=10, freq=spod.freq)

    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=f1, freq=spod.freq,
        modes_idx=[0,1,2], x1=x2, x2=x1,
        equal_axes=True, filename='modes_f1.png')

    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=f2, freq=spod.freq,
        modes_idx=[0,1,2], x1=x2, x2=x1,
        equal_axes=True, filename='modes_f2.png')
    
    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=f3, freq=spod.freq,
        modes_idx=[0,1,2], x1=x2, x2=x1,
        equal_axes=True, filename='modes_f3.png')
    
    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=f4, freq=spod.freq,
        modes_idx=[0,1,2], x1=x2, x2=x1,
        equal_axes=True, filename='modes_f4.png')
    
    ## plot 2d modes at frequency of interest
    spod.plot_2d_modes_at_frequency(freq_req=10, freq=spod.freq,
        modes_idx=[0,1,2], x1=x2, x2=x1,
        equal_axes=True, filename='modes_f5.png')

    # ## plot coefficients
    # coeffs = np.load(file_coeffs)
    # post.plot_coeffs(coeffs, coeffs_idx=[0,1],
    #     path=results_dir, filename='coeffs.png')

    # ## plot reconstruction
    # recons = np.load(file_dynamics)
    # post.plot_2d_data(recons, time_idx=[0,10], filename='recons.jpg',
    #     path=results_dir, x1=x2, x2=x1, equal_axes=True)

    # ## plot data
    # data = spod.get_data(data)
    # post.plot_2d_data(data, time_idx=[0,10], filename='data.jpg',
    #     path=results_dir, x1=x2, x2=x1, equal_axes=True)
    # post.plot_data_tracers(data, coords_list=[(5,0.5)],
    #     time_limits=[0,nt], path=results_dir, filename='data_tracers.jpg')
    # post.generate_2d_data_video(
    #     data, sampling=5, time_limits=[0,nt], x1=x2, x2=x1,
    #     path=results_dir, filename='data_movie1.mp4')
    ## -------------------------------------------------------------
