import random

import numpy as np

from simulation import Simulation
from analysis import result_analysis

np.random.seed(4800)
random.seed(4800)


if __name__ == '__main__':
    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1)
    sim.run(optimizers=['GA'], repeats=1, pareto=True)

    methods = ['GRASP', 'SA', 'GA', 'ABC', 'TABU', 'ALNS']

    sim = Simulation()
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='NO STOCHASTICITY REGULAR')

    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='MEDIUM STOCHASTICITY REGULAR')

    sim = Simulation(traffic_delay_rate=0.2, breakdown_rate=0.005, rest_delay_rate=0.3)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='HIGH STOCHASTICITY REGULAR')

    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1, num_orders=150)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='HIGH PRESSURE REGULAR')

    sim = Simulation(env_size=30, num_orders=50, num_drivers=5)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='NO STOCHASTICITY SMALL')

    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1, env_size=30, num_orders=50, num_drivers=5)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='MEDIUM STOCHASTICITY SMALL')

    sim = Simulation(traffic_delay_rate=0.2, breakdown_rate=0.005, rest_delay_rate=0.3, env_size=30, num_orders=50, num_drivers=5)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='HIGH STOCHASTICITY SMALL')

    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1, env_size=30, num_orders=75, num_drivers=5)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='HIGH PRESSURE SMALL')

    sim = Simulation(env_size=70, num_orders=150, num_drivers=15)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='NO STOCHASTICITY LARGE')

    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1, env_size=70, num_orders=150, num_drivers=15)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='MEDIUM STOCHASTICITY LARGE')

    sim = Simulation(traffic_delay_rate=0.2, breakdown_rate=0.005, rest_delay_rate=0.3, env_size=70, num_orders=150, num_drivers=15)
    sim.run(optimizers=methods, repeats=30)
    sim.analysis(title='HIGH STOCHASTICITY LARGE')

    result_analysis()


# environment 50x50 (5kmx5km)
# center 10x10 (1kmx1km)
# normal speed 3 per minute  300m/minute 18km/uur if broken speed 1 per minute