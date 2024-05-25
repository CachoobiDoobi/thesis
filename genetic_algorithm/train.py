import math
import random
import time
from datetime import datetime, timedelta

import numpy as np
from deap import base, creator, tools, algorithms
from gymnasium.spaces import Dict, MultiDiscrete, Box
from scipy.constants import c
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from carpet_simulation import CarpetSimulation
from config import param_dict
import math
import random
import time
from datetime import datetime, timedelta

import numpy as np
from deap import base, creator, tools, algorithms
from gymnasium.spaces import Dict, MultiDiscrete, Box
from scipy.constants import c
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from carpet_simulation import CarpetSimulation
from config import param_dict

n_bursts = 6

action_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'RF': MultiDiscrete(nvec=[2] * n_bursts),
     })

observation_space = Dict(
    {'pulse_duration': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'PRI': MultiDiscrete(nvec=[5] * n_bursts, start=[0] * n_bursts),
     'n_pulses': MultiDiscrete(nvec=[21] * n_bursts, start=[10] * n_bursts),
     'RF': MultiDiscrete(nvec=[2] * n_bursts),
     'PD': Box(low=0, high=1),
     'ratio': Box(low=0, high=100),
     'r_hat': Box(low=0, high=1e5),
     'v_hat': Box(low=0, high=1e3),
     'v_wind': Box(low=0, high=40),
     'alt': Box(low=10, high=30)
     }
)


# Function to initialize an individual
def initIndividual(icls):
    pulse_durations = np.random.randint(low=0, high=5, size=n_bursts)
    pri = np.random.randint(low=0, high=5, size=n_bursts)
    n_pulses = np.random.randint(low=10, high=31, size=n_bursts)
    rf = np.random.randint(low=0, high=2, size=n_bursts)

    params = np.concatenate([pulse_durations, pri, n_pulses, rf])
    return icls(params.tolist())


# Evaluation function
def evalFunction(individual, range_, velocity, altitude, rainfall_rate=2.8 * 10e-7, wind_speed=40, rcs=1):
    sim = CarpetSimulation()
    action_dict = {
        "pulse_duration": individual[:n_bursts],
        "PRI": individual[n_bursts: 2 * n_bursts],
        "n_pulses": individual[2 * n_bursts: 3 * n_bursts],
        "RF": individual[3 * n_bursts:]
    }
    try:
        pds, snr = sim.detect(action_dict, velocity=velocity, range_=range_, altitude=altitude,
                              rainfall_rate=rainfall_rate,
                              wind_speed=wind_speed, rcs=rcs)
    except ValueError:
        return (0,0)

    reward_pd = np.mean(pds)

    duration = 0
    pris = [param_dict['PRI'][pri] for pri in action_dict["PRI"]]
    n_pulses = action_dict['n_pulses']
    durations = np.multiply(pris, n_pulses)
    duration += np.sum(durations)
    min_duration = 1 / (2 * 1e9 * 30 / c)
    ratio = duration / min_duration

    sigma = 1.6
    # reward_time = math.exp(-(ratio - 1) ** 2 / (2 * sigma ** 2))  # Gaussian function
    # print(ratio, reward_time, reward_d)
    return reward_pd, ratio


def train(wind_speed=40, rcs=1, rainfall_rate=2.8 * 10e-7):
    # start_time = time.time()
    # Define the problem as a maximization problem
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    toolbox.register("individual", initIndividual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalFunction)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)

    start_time = datetime.now()

    transition_model_altitude = CombinedLinearGaussianTransitionModel([ConstantVelocity(100)])

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1)])

    truth_alt = GroundTruthPath(
        [GroundTruthState([np.random.uniform(10, 30), np.random.uniform(1, 3)], timestamp=start_time)])

    # 1d model
    truth = GroundTruthPath(
        [GroundTruthState([np.random.uniform(1e4, 5e4), np.random.uniform(100, 500)], timestamp=start_time)])

    for k in range(1, 21):
        truth.append(GroundTruthState(
            transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k)))
        truth_alt.append(GroundTruthState(
            transition_model_altitude.function(truth_alt[k - 1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k)))
        # print(truth[k].state_vector[0], truth[k].state_vector[1])


    pds = []
    ratios = []
    start_time = time.time()
    for i in range(20):
                range_ = truth[i].state_vector[0]
                velocity = truth[i].state_vector[1]
                alt = truth_alt[i].state_vector[0]
                altitude = alt if alt > 0 else abs(alt)
                # Create the population
                pop = toolbox.population(n=300)

                # Define the number of generations
                ngen = 50
                # Define the probability of mating two individuals
                cxpb = 0.5
                # Define the probability of mutating an individual
                mutpb = 0.2

                # Redefine evaluation function with fixed range, velocity, and altitude
                evalFunctionFixed = lambda ind: evalFunction(ind, range_=range_, velocity=velocity,
                                                             altitude=altitude, wind_speed=wind_speed, rcs=rcs, rainfall_rate=rainfall_rate)
                toolbox.register("evaluate", evalFunctionFixed)

                # Run the genetic algorithm
                algorithms.eaSimple(pop, toolbox, cxpb, mutpb, ngen, verbose=False)

                # Get the best individual
                best_ind = tools.selBest(pop, 1)[0]
                # print(
                #     f"For range={range_}, velocity={velocity}, altitude={altitude}, best individual is {best_ind} with fitness {best_ind.fitness.values}")
                pds.append(best_ind.fitness.values[0])
                ratios.append(best_ind.fitness.values[1])

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Elapsed time: {elapsed_time} seconds')
    return pds, ratios

if __name__ == "__main__":
    train()
