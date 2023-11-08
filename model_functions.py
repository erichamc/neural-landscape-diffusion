from collections import namedtuple
from jax import random, grad, jit, lax, vmap
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal
from jax.scipy.stats import t as studentst
import optax
import functools
import seaborn as sns
import scipy
import sys
import os
import matplotlib
from matplotlib import pyplot as plt

##########################
### FIXED MODEL PARAMS ###
##########################

dt = 1./100
water_feedback_constant = 0.006
food_feedback_constant = 0.004
feedback_delay = 120 * 100
thirst_mean = np.array([5.,7.5])
hunger_mean = np.array([5.,-7.5])
other_mean = np.array([-8, 0.])
pdf_var = 20.

##########################
###  MODEL FUNCTIONS #####
##########################

# Regular state tuple
fState = namedtuple("State", ["u", "rng_key", "all_u", "thirst", "hunger", "go", "choice", "need", "params"])
# Opto-stim state tuple
fiState = namedtuple("State", ["u", "rng_key", "all_u", "thirst", "hunger", "go", "choice", "need", "params", "thirst_stim"])
# Forced-transitions alternative model state tuple
forceState = namedtuple("State", ["u", "rng_key", "all_u", "thirst", "hunger", "go", "choice", "need", "params", "impulse"])

thirst_pdf = lambda x: multivariate_normal.logpdf(x, jnp.array(thirst_mean), pdf_var*jnp.eye(2))
hunger_pdf = lambda x: multivariate_normal.logpdf(x, jnp.array(hunger_mean), pdf_var*jnp.eye(2))
other_pdf = lambda x: multivariate_normal.logpdf(x, jnp.array(other_mean), pdf_var*jnp.eye(2))

def miss(x): return other_pdf(x) >= jnp.logaddexp(thirst_pdf(x),hunger_pdf(x))
def water_choice(x): return jnp.logical_and(~miss(x), thirst_pdf(x) >= hunger_pdf(x))
def food_choice(x): return jnp.logical_and(~miss(x), ~water_choice(x))
def reward_choice(x): return thirst_pdf(x) > hunger_pdf(x)

# LANDSCAPE EQUATION
@jit
def f(x, t=3, h=3, params=jnp.array([1., 1., 1., 1.])):
    # Define the energy landscape, shaped by thirst and hunger magnitudes
    
    # force t, h inputs to be positive
    t = jnp.max(jnp.array([0, t]))
    h = jnp.max(jnp.array([0, h]))
    # sum up functions
    return -1*params[3]*jnp.logaddexp(jnp.logaddexp(jnp.log(params[2]*t)+thirst_pdf(x),
                                                    jnp.log(params[2]*h)+hunger_pdf(x)
                                     ), other_pdf(x))

##########################
### UPDATE EQUATIONS #####
##########################

# UPDATE EQUATION / EQUATION OF MOTION
@jit
def step_f(i, state):
    # extract state
    u, rng_key, all_u, thirst, hunger, go, choice, need, par = state
    rng_key, noise_key = random.split(rng_key, 2)
    
    hit_ = ~miss(u)
    water_ = reward_choice(u)
    hit = jnp.logical_and(go[i]==1, hit_)
    water = jnp.logical_and(hit, water_)
    food = jnp.logical_and(hit, ~water_)
    # current choice
    choice = choice.at[i].set(jnp.array([hit, food, water, go[i]==2]))
    
    # delayed feedback
    didx = jnp.max(jnp.array([0, i-feedback_delay]))
    delayed_u = all_u[didx]
    dhit_ = ~miss(delayed_u)
    dwater_ = reward_choice(delayed_u)
    hit = jnp.logical_and(go[didx]==1, dhit_)
    dwater = jnp.logical_and(hit, dwater_)
    dfood = jnp.logical_and(hit, ~dwater_)
    thirst = jnp.max(jnp.array([thirst - par['water_feedback_constant']*dwater, 0.01]))
    hunger = jnp.max(jnp.array([hunger - par['food_feedback_constant']*dfood, 0.01]))
    
    need = need.at[i].set(jnp.array([thirst, hunger]))
    
    grad_potential = -1*grad(f, argnums=0)(u, thirst, hunger, jnp.array([1., 1., par['needs_weight'], par["well_scale"]]))
    # equation of motion
    u_new = u+(dt*par['gradient_weight']*grad_potential+jnp.sqrt(dt)*par['noise_weight']*(random.multivariate_normal(noise_key, u, jnp.eye(2))-u))
    
    all_u = all_u.at[i].set(u_new)
    return fState(u_new, rng_key, all_u, thirst, hunger, go, choice, need, par)

# UPDATE FUNCTION FOR SIMULATING OPTOGENETIC STIMULATION
@jit
def step_f_opto(i, state):
    # extract state
    u, rng_key, all_u, thirst, hunger, go, choice, need, par, thirst_stim = state
    rng_key, noise_key = random.split(rng_key, 2)
    
    hit_ = ~miss(u)
    water_ = reward_choice(u)
    hit = jnp.logical_and(go[i]==1, hit_)
    water = jnp.logical_and(hit, water_)
    food = jnp.logical_and(hit, ~water_)
    # current choice
    choice = choice.at[i].set(jnp.array([hit, food, water, go[i]==2]))
    
    # delayed feedback
    didx = jnp.max(jnp.array([0, i-feedback_delay]))
    delayed_u = all_u[didx]
    dhit_ = ~miss(delayed_u)
    dwater_ = reward_choice(delayed_u)
    hit = jnp.logical_and(go[didx]==1, dhit_)
    dwater = jnp.logical_and(hit, dwater_)
    dfood = jnp.logical_and(hit, ~dwater_)
    thirst = jnp.max(jnp.array([thirst - par['water_feedback_constant']*dwater, 0.01]))
    hunger = jnp.max(jnp.array([hunger - par['food_feedback_constant']*dfood, 0.01]))
    
    need = need.at[i].set(jnp.array([thirst+thirst_stim[i], hunger]))
    
    grad_potential = -1*grad(f, argnums=0)(u, thirst+thirst_stim[i], hunger, jnp.array([1., 1., par['needs_weight'], par['well_scale']]))
    # equation of motion
    u_new = u+(dt*par['gradient_weight']*grad_potential+jnp.sqrt(dt)*par['noise_weight']*(random.multivariate_normal(noise_key, u, jnp.eye(2))-u))
    all_u = all_u.at[i].set(u_new)
    return fiState(u_new, rng_key, all_u, thirst, hunger, go, choice, need, par, thirst_stim)

# UPDATE EQUATION FOR SIMULATING FORCED-TRANSITIONS ALTERNATIVE MODEL
@jit
def step_forced(i, state):
    # extract state
    u, rng_key, all_u, thirst, hunger, go, choice, need, par, impulse = state
    rng_key, noise_key = random.split(rng_key, 2)
    
    hit_ = ~miss(u)
    water_ = reward_choice(u)
    hit = jnp.logical_and(go[i]==1, hit_)
    water = jnp.logical_and(hit, water_)
    food = jnp.logical_and(hit, ~water_)
    # current choice
    choice = choice.at[i].set(jnp.array([hit, food, water, go[i]==2]))
    
    # delayed feedback
    didx = jnp.max(jnp.array([0, i-feedback_delay]))
    delayed_u = all_u[didx]
    dhit_ = ~miss(delayed_u)
    dwater_ = reward_choice(delayed_u)
    hit = jnp.logical_and(go[didx]==1, dhit_)
    dwater = jnp.logical_and(hit, dwater_)
    dfood = jnp.logical_and(hit, ~dwater_)
    thirst = jnp.max(jnp.array([thirst - par['water_feedback_constant']*dwater, 0.01]))
    hunger = jnp.max(jnp.array([hunger - par['food_feedback_constant']*dfood, 0.01]))
    
    fduration = jnp.array(2/dt).astype('int32')
    push = jnp.array([0,-1*jnp.sign(all_u[i-fduration,1])*impulse[i] * .15])
    
    need = need.at[i].set(jnp.array([thirst, hunger]))
    
    grad_potential = grad(f, argnums=0)(u, thirst, hunger, jnp.array([1., 1., par['needs_weight'], par['well_scale']]))
    # equation of motion
    u_new = u+(dt*grad_potential*par['gradient_weight']+jnp.sqrt(dt)*par['noise_weight']*(random.multivariate_normal(noise_key, u, jnp.eye(2))-u))+dt*push
    all_u = all_u.at[i].set(u_new)
    return forceState(u_new, rng_key, all_u, thirst, hunger, go, choice, need, par, impulse)


##########################
## SIMULATION FUNCTIONS ##
##########################

def get_bounded_trial_times(nsamples, rkey, dt=1/100, min_trial_time=4.1):
    ntrials = int(nsamples / (min_trial_time/dt))
    #iti = jnp.min(jnp.stack(jnp.array([random.exponential(rkey, jnp.array([ntrials]))*3, jnp.ones(ntrials)*5])).T, axis=1)
    iti = random.uniform(rkey, jnp.array([ntrials]), minval=3, maxval=6)
    trial_times = jnp.cumsum(min_trial_time + iti)
    go_trials = random.choice(rkey, jnp.array([0,1]), (ntrials,), p=jnp.array([1/3, 2/3])).astype('bool')
    valid_trial_idx = trial_times < (nsamples*dt)
    go_trials = go_trials[valid_trial_idx]
    nogo_trials = ~go_trials
    gotrial_times = trial_times[valid_trial_idx][go_trials]
    nogo_trial_times = trial_times[valid_trial_idx][nogo_trials]
    trial_idx = (gotrial_times/dt).astype('int32')
    nogo_idx = (nogo_trial_times/dt).astype('int32')
    go_array = jnp.zeros(nsamples)
    go_array = go_array.at[trial_idx].set(1)
    go_array = go_array.at[nogo_idx].set(2)
    return go_array

def get_random_series(ix, nsamples, nsim):
    sim_start_key = random.PRNGKey(ix) # reproduce simulation randomness
    rng_key, *all_keys = random.split(sim_start_key, nsim+1)
    all_keys = jnp.array(all_keys)
    all_go_array = jnp.stack([get_bounded_trial_times(nsamples, all_keys[i], dt) for i in range(nsim)])
    return all_keys, all_go_array


# Run forward simulation, picking random initial thirst and hunger on the given interval
@jit
def run_simulation(k, go_array, p, min_need=0.8, max_need=2.):
    ''' Run a forward simulation, using uniform random initial thirst and hunger.
        Inputs: 
            k: a jax random key
            go_array: trial array of length # samples, 1 if go, 2 if no-go
            p: parameter dictionary to use for simulation
                {"gradient_weight": #
                 "needs_weight": #,
                 "noise_weight": sqrt(2*KbT),
                 "water_feedback_constant": #
                 "food_feedback_constant": ##}
            min_need: lower limit of uniform distribution for need
            max_need: upper limit of uniform distribution for need
    '''
    # split random key
    k, subkey = random.split(k)
    initial_thirst = random.uniform(subkey,minval=min_need, maxval=max_need)
    #initial_thirst = random.uniform(subkey,minval=0.1, maxval=3.)
    k, subkey = random.split(k)
    initial_hunger = random.uniform(subkey,minval=min_need, maxval=max_need)
    #initial_hunger = random.uniform(subkey,minval=0.1, maxval=3.)
    x0 = random.choice(k, jnp.array([hunger_mean, thirst_mean])) # randomly pick food or water as start
    nsamples = go_array.shape[0]
    samples =jnp.zeros((nsamples, 2))
    choice = jnp.zeros((nsamples,4))
    need = jnp.zeros((nsamples, 2))
    state = fState(x0, subkey, samples, initial_thirst, initial_hunger, go_array, choice, need, p)
    return lax.fori_loop(0,nsamples, step_f, state)

# Run forward simulation, given initial thirst and hunger
@jit
def run_simulation_(k, go_array, p, initial_thirst, initial_hunger):
    ''' Run a forward simulation, using uniform random initial thirst and hunger.
        Inputs: 
            k: a jax random key
            go_array: trial array of length # samples, 1 if go, 2 if no-go
            p: parameter dictionary to use for simulation
                {"dyanmics_weight": alpha,
                 "noise_weight": sqrt(2*KbT),
                 "water_feedback_constant": #
                 "food_feedback_constant": ##}
            initial_thirst: starting thirst magnitude value
            initial_hunger: starting hunger matgnitude value
    '''
    # split random key
    k, subkey = random.split(k)
    nsamples = go_array.shape[0]
    samples =jnp.zeros((nsamples, 2))
    choice = jnp.zeros((nsamples,4))
    need = jnp.zeros((nsamples, 2))
    x0 = random.choice(k, jnp.array([hunger_mean, thirst_mean]))
    state = fState(x0, subkey, samples, initial_thirst, initial_hunger, go_array, choice, need, p)
    return lax.fori_loop(0,nsamples, step_f, state)

# Run forward simulation, given initial hunger and thirst, and optogenetic stimulation intensity
# Note that the optogenetic input timing is currently hardcoded within the function
@jit
def run_simulation_inputs(k, go_array, p, initial_thirst, initial_hunger, stim_intensity):
    # split random key
    k, subkey = random.split(k)
    nsamples = go_array.shape[0]
    samples = jnp.zeros((nsamples, 2))
    choice = jnp.zeros((nsamples,4))
    need = jnp.zeros((nsamples, 2))
    pulses = jnp.tile(jnp.concatenate([jnp.zeros(int(110/dt)),stim_intensity*jnp.ones(int(10/dt))]), 25)
    opto = jnp.concatenate([pulses, jnp.zeros(nsamples-len(pulses))])
    pw0 = initial_thirst / (initial_thirst + initial_hunger)
    pf0 = 1-pw0
    x0 = random.choice(k, jnp.array([hunger_mean, thirst_mean]), p=jnp.array([pf0, pw0]))
    #x0 = hunger_mean
    state = fiState(jnp.array([5.,0.]), subkey, samples, initial_thirst, initial_hunger, go_array, choice, need, p, opto)
    return lax.fori_loop(0,nsamples, step_f_opto, state)

# Run forward simulation for "forced transition" alternative model
def run_forced_simulation(k, go_array, p, initial_thirst, initial_hunger, impulse):
    # split random key
    k, subkey = random.split(k)
    nsamples = go_array.shape[0]
    samples = jnp.zeros((nsamples, 2))
    choice = jnp.zeros((nsamples,4))
    need = jnp.zeros((nsamples, 2))
    state = forceState(jnp.array([5.,0.]), subkey, samples, initial_thirst, initial_hunger, go_array, choice, need, p, impulse)
    return lax.fori_loop(0,nsamples, step_forced, state)

##########################
### THEORETICAL FUNCS  ###
##########################

### Set some constants ###

# Points along the transition path between wells
transition_path = jnp.array(np.linspace(jnp.array([hunger_mean[0],
                                                   hunger_mean[1]+2.5]),
                                        jnp.array([thirst_mean[0],
                                                   thirst_mean[1]-2.5]),1000))

# Sample points around the occupied space of the three states
minval = -25
maxval = 25
sample_density = 400
X,Y = jnp.meshgrid(jnp.linspace(minval,maxval,sample_density), jnp.linspace(minval,maxval,sample_density))
points = jnp.vstack([X.flatten(), Y.flatten()]).T
points = points[(thirst_pdf(points) > jnp.percentile(thirst_pdf(points), 67)) |
                (hunger_pdf(points) > jnp.percentile(hunger_pdf(points), 67)) |
                (other_pdf(points) > jnp.percentile(other_pdf(points), 67))]
water_points = points[water_choice(points)]
food_points = points[food_choice(points)]
other_points = points[miss(points)]

# Minimal value for numerical stability
eps = jnp.finfo('float32').eps

@jit
def grad_f(x, t, h, params):
    '''Jitted wrapper for gradient'''
    return grad(f, argnums=0)(x, t, h, params)[1]

@jit
def ts_x(t,h,params):
    '''Return the single transition point coordinate'''
    ix = jnp.argmax(f(transition_path, t=t, h=h, params=params))
    return transition_path[ix]

# 
@jit
def E_ts(t,h,params):
    '''Transition state energy'''
    return jnp.max(f(transition_path, t, h, params))
    
@jit
def Wwf(t,h,params):
    return ((jnp.sqrt(1/pdf_var)*jnp.sqrt(2/pdf_var))/(2*jnp.pi*params[0]))*jnp.exp(-1*(E_ts(t,h,params)-f(thirst_mean,t=t,h=h,params=params))/params[1])
    
@jit
def Wfw(t,h,params):
    return ((jnp.sqrt(1/pdf_var)*jnp.sqrt(2/pdf_var))/(2*jnp.pi*params[0]))*jnp.exp(-1*(E_ts(t,h,params)-f(hunger_mean,t=t,h=h,params=params))/params[1])

@jit
def req_a_to_b(t,h,params):
    return Wwf(t,h,params)/(Wwf(t,h,params)+Wfw(t,h,params))

@jit
def req_b_to_a(t,h,params):
    return Wfw(t,h,params)/(Wwf(t,h,params)+Wfw(t,h,params))

@jit
def Pww(t,h,time,params):
    # Transition from water to water
    return (1-req_b_to_a(t,h,params))*jnp.exp(-1*(Wfw(t,h,params)+Wwf(t,h,params))*(time))+req_b_to_a(t,h,params)

@jit
def Pwf(t,h,time,params):
    # Transition from water to food
    return 1-Pww(t,h,time,params)

@jit
def Pff(t,h,time,params):
    # Transition from food to food
    return (1-req_a_to_b(t,h,params))*jnp.exp(-1*(Wfw(t,h,params)+Wwf(t,h,params))*(time))+req_a_to_b(t,h,params)

@jit
def Pfw(t,h,time,params):
    # Transition from food to water
    return 1-Pff(t,h,time,params)

@jit
def marginal_test(t,h,params):
    # P(water | t, h, params) independent of history
    
    w_int = jnp.sum(jnp.exp((-1/params[1])*params[3]*f(water_points, t, h, params)))
    f_int = jnp.sum(jnp.exp((-1/params[1])*params[3]*f(food_points, t, h, params)))
    o_int = jnp.sum(jnp.exp((-1/params[1])*params[3]*f(other_points, t, h, params)))
    
    return w_int / (w_int+f_int+o_int)


@jit
def marginal_water(t,h,params):
    ''' P(water | t, h, params) independent of history. Ignoring other state.'''
    
    w_int = jnp.sum(jnp.exp((-1/params[1])*f(water_points, t, h, params)))
    f_int = jnp.sum(jnp.exp((-1/params[1])*f(food_points, t, h, params)))
    
    return w_int / (w_int+f_int)

@jit
def marginal_food(t,h,params):
    ''' P(food | t, h, params) independent of history. Ignoring other state. '''
    
    w_int = jnp.sum(jnp.exp((-1/params[1])*f(water_points, t, h, params)))
    f_int = jnp.sum(jnp.exp((-1/params[1])*f(food_points, t, h, params)))
    
    return f_int / (w_int+f_int)
    
@jit
def marginal_water_3s(t,h,params):
    ''' P(water | t, h, params) independent of history '''
    
    w_int = jnp.sum(jnp.exp((-1/params[1])*f(water_points, t, h, params)))
    f_int = jnp.sum(jnp.exp((-1/params[1])*f(food_points, t, h, params)))
    o_int = jnp.sum(jnp.exp((-1/params[1])*f(other_points, t, h, params)))
    
    return w_int / (w_int+f_int+o_int)

@jit
def marginal_food_3s(t,h,params):
    ''' P(food | t, h, params) independent of history'''
    
    w_int = jnp.sum(jnp.exp((-1/params[1])*f(water_points, t, h, params)))
    f_int = jnp.sum(jnp.exp((-1/params[1])*f(food_points, t, h, params)))
    o_int = jnp.sum(jnp.exp((-1/params[1])*f(other_points, t, h, params)))
    
    return f_int / (w_int+f_int+o_int)
    
@jit
def marginal_other(t,h,params):
    ''' P(miss | t, h, params) independent of history '''
    
    w_int = jnp.sum(jnp.exp((-1/params[1])*f(water_points, t, h, params)))
    f_int = jnp.sum(jnp.exp((-1/params[1])*f(food_points, t, h, params)))
    o_int = jnp.sum(jnp.exp((-1/params[1])*f(other_points, t, h, params)))
    return o_int / (w_int + f_int + o_int)

@jit
def norm_need(t,h):
    return (t-h)/(0.0001+t+h)

t_bins = np.arange(0.1,1.5,0.3)
h_bins = np.arange(0.1,1.5,0.3)
thirsts = np.stack(np.meshgrid(t_bins,h_bins),-1)[...,0].flatten()
hungers = np.stack(np.meshgrid(t_bins,h_bins),-1)[...,1].flatten()

##########################
### LOSS FUNCTIONS #######
##########################

@functools.partial(jax.vmap, in_axes=(None, 0))
@jit
def loss_fw_boltzmann(params, x):
    '''Boltzmann negative log probability of a given trial outcome'''
    # Excluding misses.
    # x: [outcome {0: food, 1: water}, thirst, hunger]

    probs = jnp.array([marginal_food(x[1], x[2], params), marginal_water(x[1], x[2], params)])

    return -1*jnp.log(probs[x[0].astype('int32')])


@functools.partial(jax.vmap, in_axes=(None, 0))
@jit
def loss_miss_boltzmann(params, x):
    '''Boltzmann negative log probability of a given trial outcome'''
    # x: [outcome {0: food, 1: water, 2: miss}, thirst, hunger]

    probs = jnp.array([marginal_food_3s(x[1], x[2], params), marginal_water_3s(x[1], x[2], params), marginal_other(x[1], x[2], params)])

    return -1*jnp.log(probs[x[0].astype('int32')])

     
@functools.partial(jax.vmap, in_axes=(None, 0))
@jit
def nll(params, x):
    ''' Negative log likelihood of a single trial'''
    tPa =  Pww(x[2], x[3], x[4], params)
    tPb =  Pff(x[2], x[3], x[4], params)
    transitions = -1*jnp.log(eps+jnp.array([[tPb, 1-tPb], [1-tPa, tPa]]))
    outcome_logprob = transitions[x[0].astype('int32'), x[1].astype('int32')]
    
    return outcome_logprob

@jit
def loss_joint(params, trials, marginals):
    '''Joint loss function combining nll (per trial) and marginal (boltzmann) loss'''
    
    # Pinning friction term to 1.
    params = jnp.hstack([jnp.array([1.0]), jnp.array([params[1]]), jnp.array([3.]), jnp.array(params[3])])
    # Combine per-trial average loss
    return jnp.nanmean(nll(params,trials)) + jnp.nanmean(loss_fw_boltzmann(params, marginals))

@jit
def loss_sated(need_scale, params, sated):
    ''' Boltzmann state probability for fitting foraging scale using sated data'''
    params = params.at[2].set(need_scale)
    return jnp.nanmean(loss_miss_boltzmann(params, sated)) + 0.001*need_scale


##########################
### PLOTTING UTILITIES ###
##########################

def plot_simulated_behavior(fstate, key=None, dpi=200):
    col = dict(zip(["water","food","miss"], [sns.color_palette()[i] for i in [0,1,2]]))
    trial_encoding = []
    t = fstate.choice[fstate.go>0,:]
    temp = np.array(t).copy()
    temp[:,1] = temp[:,1]*1
    temp[:,2] = temp[:,2]*2
    temp[:,3] = temp[:,3]*3
    temp[:,0] = -1*((temp[:,0]==0)&(temp[:,3]==0))
    temp[temp[:,0]==-1,0] = 4
    temp = temp.sum(1)-1
    trial_encoding.append(temp)
    trial_encoding = np.array(trial_encoding[0])
    plt.figure(figsize=(1.5,3), dpi=dpi)
    
    stop_threshold = 40#np.random.uniform(20,50)
        
    ax1 = plt.subplot(111)
    
    go_count = 0
    contiguous_miss = 0
    max_trial = len(trial_encoding[trial_encoding!=2])+20

    for j,t in enumerate(trial_encoding[trial_encoding!=2]):
        curr_licks = np.random.uniform(0.3,0.7)+np.arange(0,np.random.uniform(0.7,1.5),np.random.uniform(0.1,0.2))
        if t==0:
            ax1.plot(curr_licks,j*np.ones((len(curr_licks),)),'.', color=col['food'], markersize=2, alpha=0.5, rasterized=True)
            go_count = j
            contiguous_miss = 0
        elif t==1:
            ax1.plot(curr_licks,j*np.ones((len(curr_licks),)),'.', color=col['water'], markersize=2, alpha=0.5, rasterized=True)
            go_count = j
            contiguous_miss = 0
        else:
            contiguous_miss += 1
            if contiguous_miss > stop_threshold:
                max_trial = j
                break

    ax1.axvline(0,color='k', linestyle='--')       
    ax1.set_ylim(max_trial,0)
    ax1.set_xlim([-1.1, 3.5])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Go trial')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax1.axhspan(go_count, max_trial, color='k', alpha=0.2)
    
    if key is not None:
        plt.savefig(get_dropbox_path()+f"/buridan_figures/model/revision_model_sim_behavior_key{key}.pdf",
                transparent=True,
                bbox_inches='tight')

def plot_simulated_behavior2(fstate, key=None):
    col = dict(zip(["water","food","miss"], [sns.color_palette()[i] for i in [0,1,2]]))
    trial_encoding = []
    t = fstate.choice[fstate.go>0,:]
    temp = np.array(t).copy()
    temp[:,1] = temp[:,1]*1
    temp[:,2] = temp[:,2]*2
    temp[:,3] = temp[:,3]*3
    temp[:,0] = -1*((temp[:,0]==0)&(temp[:,3]==0))
    temp[temp[:,0]==-1,0] = 4
    temp = temp.sum(1)-1
    trial_encoding.append(temp)
    trial_encoding = np.array(trial_encoding[0])
    plt.figure(figsize=(1.5,3))
    
    ax1 = plt.subplot(111)
    
    go_count = 0
    max_trial = len(trial_encoding[trial_encoding!=2])

    for j,t in enumerate(trial_encoding[trial_encoding!=2]):
        curr_licks = np.random.uniform(0.3,0.7)+np.arange(0,np.random.uniform(0.7,1.5),np.random.uniform(0.1,0.2))
        if t==0:
            ax1.plot(curr_licks,j*np.ones((len(curr_licks),)),'.', color=col['food'], markersize=2, alpha=0.5, rasterized=True)
            go_count = j
        elif t==1:
            ax1.plot(curr_licks,j*np.ones((len(curr_licks),)),'.', color=col['water'], markersize=2, alpha=0.5, rasterized=True)
            go_count = j
        else:
            pass

    ax1.axvline(0,color='k', linestyle='--')       
    ax1.set_ylim(max_trial,0)
    ax1.set_xlim([-1.1, 3.5])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Go trial')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
        
