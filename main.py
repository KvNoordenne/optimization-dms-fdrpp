import random

import numpy as np

from simulation import Simulation
from analysis import result_analysis

np.random.seed(4800)
random.seed(4800)


if __name__ == '__main__':
    # run a simulation of GA to analyze pareto-front
    sim = Simulation(traffic_delay_rate=0.1, breakdown_rate=0.002, rest_delay_rate=0.1)
    sim.run(optimizers=['GA'], repeats=1, pareto=True)

    # set global methods
    methods = ['GRASP', 'SA', 'GA', 'ABC', 'TABU', 'ALNS']

    # set configurations for each environment
    scenarios = [
        {"title": "NO STOCHASTICITY REGULAR", "config": {}},
        {"title": "MEDIUM STOCHASTICITY REGULAR", "config": {"traffic_delay_rate": 0.1, "breakdown_rate": 0.002, "rest_delay_rate": 0.1}},
        {"title": "HIGH STOCHASTICITY REGULAR", "config": {"traffic_delay_rate": 0.2, "breakdown_rate": 0.005, "rest_delay_rate": 0.3}},
        {"title": "HIGH PRESSURE REGULAR", "config": {"traffic_delay_rate": 0.1, "breakdown_rate": 0.002, "rest_delay_rate": 0.1, "num_orders": 150}},

        {"title": "NO STOCHASTICITY SMALL", "config": {"env_size": 30, "num_orders": 50, "num_drivers": 5}},
        {"title": "MEDIUM STOCHASTICITY SMALL", "config": {"traffic_delay_rate": 0.1, "breakdown_rate": 0.002, "rest_delay_rate": 0.1, "env_size": 30, "num_orders": 50, "num_drivers": 5}},
        {"title": "HIGH STOCHASTICITY SMALL", "config": {"traffic_delay_rate": 0.2, "breakdown_rate": 0.005, "rest_delay_rate": 0.3, "env_size": 30, "num_orders": 50, "num_drivers": 5}},
        {"title": "HIGH PRESSURE SMALL", "config": {"traffic_delay_rate": 0.1, "breakdown_rate": 0.002, "rest_delay_rate": 0.1, "env_size": 30, "num_orders": 75, "num_drivers": 5}},

        {"title": "NO STOCHASTICITY LARGE", "config": {"env_size": 70, "num_orders": 150, "num_drivers": 15}},
        {"title": "MEDIUM STOCHASTICITY LARGE", "config": {"traffic_delay_rate": 0.1, "breakdown_rate": 0.002, "rest_delay_rate": 0.1, "env_size": 70, "num_orders": 150, "num_drivers": 15}},
        {"title": "HIGH STOCHASTICITY LARGE", "config": {"traffic_delay_rate": 0.2, "breakdown_rate": 0.005, "rest_delay_rate": 0.3, "env_size": 70, "num_orders": 150, "num_drivers": 15}}
    ]

    # run all experiments
    for item in scenarios:
        print(f"Starting Scenario Execution: {item['title']}...")

        sim = Simulation(**item['config'])
        sim.run(optimizers=methods, repeats=30)
        sim.analysis(title=item['title'])

    result_analysis()


# environment 50x50 (5kmx5km)
# center 10x10 (1kmx1km)
# normal speed 3 per minute  300m/minute 18km/uur if broken speed 1 per minute