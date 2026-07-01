import random
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from environment_classes import Environment, Order, Stop


class Simulation:
    def __init__(self, num_drivers=10, num_orders=100, alpha=0.11, beta=0.89, traffic_delay_rate=0.0, rest_delay_rate=0.0,
                 breakdown_rate=0.0, t_max=240, env_size=50):

        self.env = None
        self.num_drivers = num_drivers
        self.num_orders = num_orders
        self.rest_delay_rate = rest_delay_rate
        self.traffic_delay_rate = traffic_delay_rate
        self.breakdown_rate = breakdown_rate
        self.alpha = alpha
        self.beta = beta
        self.t_max = t_max
        self.log = {}
        self.run_x = None
        self.env_size = env_size

        self.order_events = None
        self.traffic_delays = None
        self.breakdowns = None

        self.driver_costs = 0
        self.customer_costs = 0
        self.total_costs = 0

    def create_order(self, time, ids):
        xp = random.randint(round(self.env_size * 0.4), round(self.env_size * 0.6))
        yp = random.randint(round(self.env_size * 0.4), round(self.env_size * 0.6))
        xd = random.randint(0, self.env_size)
        yd = random.randint(0, self.env_size)
        while xd == xp and yd == yp:
            xd = random.randint(0, self.env_size)
            yd = random.randint(0, self.env_size)
        pickup = Stop(xp, yp, 'PICKUP')
        delivery = Stop(xd, yd, 'DELIVERY')
        placement = time
        ept = time + 15 # expected prepared time
        apt = time + max(10, round(random.gauss(0, 15 * self.rest_delay_rate))) # actual prepared time
        order = Order(ids, pickup, delivery,'HIDDEN', ept, apt)
        pickup.order = order
        delivery.order = order
        return order, placement

    def initialize_orders(self, visualize=False):
        orders = []
        for j in range(round(self.num_orders * 0.2)): # generate 20 orders in 0-60 min
            orders.append(self.create_order(time=random.randint(0, round(self.t_max * 0.25)), ids=j))
        i = round(self.num_orders * 0.2)
        for j in range(round(self.num_orders * 0.35)): # generate 35 orders in 61-120 min
            orders.append(self.create_order(time=random.randint(round(self.t_max * 0.25) + 1, round(self.t_max * 0.5)), ids=i + j))
        i += round(self.num_orders * 0.35)
        for j in range(round(self.num_orders * 0.35)): # generate 35 orders in 121-180 min
            orders.append(self.create_order(time=random.randint(round(self.t_max * 0.5) + 1, round(self.t_max * 0.75)), ids=i + j))
        i += round(self.num_orders * 0.35)
        for j in range(round(self.num_orders * 0.1)): # generate 10 orders in 181-210 min
            orders.append(self.create_order(time=random.randint(round(self.t_max * 0.75) + 1 , round(self.t_max * 0.875)), ids=i + j))
        orders.sort(key=lambda x: x[1])
        # print("orders", orders)
        if visualize:
            xp, yp, xd, yd = [], [], [], []
            for order in orders:
                xp.append(order[0].pickup_location.x)
                yp.append(order[0].pickup_location.y)
                xd.append(order[0].delivery_location.x)
                yd.append(order[0].delivery_location.y)
            xp = np.array(xp)
            yp = np.array(yp)
            xd = np.array(xd)
            yd = np.array(yd)

            plt.plot(xp, yp, 'ro')
            plt.plot(xd, yd, 'bo')
            plt.grid()
            plt.xlim(0, self.env_size)
            plt.ylim(0, self.env_size)
            plt.show()

        print('NEW ORDERS', orders)
        log_orders = [str(order) for order in orders]
        self.log[self.run_x]['ORDERS'] = log_orders
        return orders

    def initialize_traffic_delays(self):
        traffic_delays = {}

        for driver in range(self.num_drivers):
            traffic_delays[driver] = []

        delay_ids = [random.randint(0, self.num_drivers-1) for _ in range(round(self.t_max * 2 * self.traffic_delay_rate * self.num_drivers))]

        for driver in delay_ids:
            traffic_delays[driver].append(random.randint(0, round(self.t_max * 2)-1))
            traffic_delays[driver].sort()

        # for driver in range(self.num_drivers):
        #     delays = [random.randint(0, round(self.t_max * 1.5)-1) for _ in range(round(self.t_max * 1.5 * self.traffic_delay_rate))]
        #     traffic_delays[driver] = sorted(delays)
        print('NEW DELAYS', traffic_delays)
        self.log[self.run_x]['TRAFFIC_DELAYS'] = traffic_delays
        return traffic_delays

    def initialize_breakdowns(self):
        breakdowns = {}

        for driver in range(self.num_drivers):
            breakdowns[driver] = []

        breakdown_ids = [random.randint(0, self.num_drivers-1) for _ in range(round(self.t_max * 2 * self.breakdown_rate * self.num_drivers))]

        for breakdown in breakdown_ids:
            breakdowns[breakdown].append(random.randint(0, round(self.t_max * 2)-1))
            breakdowns[breakdown].sort()


        # for driver in range(self.num_drivers):
        #     events = [random.randint(0, round(self.t_max * 0.5)-1) for _ in range(round(self.t_max * 1.5 * self.breakdown_rate))]
        #     breakdowns[driver] = sorted(events)
        print('NEW BREAKDOWNS', breakdowns)
        self.log[self.run_x]['BREAKDOWNS'] = breakdowns
        return breakdowns

    def set_stochasticity(self):
        # print('PRESET', self.env.order_events)
        self.env.order_events = self.order_events.copy()
        self.env.traffic_delays = self.traffic_delays.copy()
        self.env.breakdowns = self.breakdowns.copy()
        # print('AFTER', self.env.order_events)

    def driver_constraint_validation(self, driver=None, route=None, inventory=None, assigned=None):
        if driver is not None:
            route = driver.route.copy()
            inventory_cap = driver.inventory.copy()
            inventory_loc = driver.inventory.copy()
            assigned = driver.assigned.copy()
        else:
            route = route.copy()
            inventory_cap = inventory.copy()
            inventory_loc = inventory.copy()
            assigned = assigned.copy()

        if route is not None and inventory_cap is not None and assigned is not None:
            # check capacity constraints
            for location in route:
                if location.type == 'PICKUP':
                    inventory_cap.append(location.order)
                    if len(inventory_cap) > self.env.capacity_constraint:
                        # print('CONSTRAINT FAIL ID 1')
                        return False, 1
                elif location.type == 'DELIVERY':
                    if location.order in inventory_cap:
                        inventory_cap.remove(location.order)
                    else:
                        # print('CONSTRAINT FAIL ID 2')
                        return False, 2
            # check for missed deliveries
            if inventory_cap:
                # print('CONSTRAINT FAIL ID 3')
                return False, 3

            # check if assigned orders have all location
            for order in assigned:
                if order in inventory_loc:
                    if order.pickup_location in route:
                        # print('CONSTRAINT FAIL ID 4')
                        return False, 4
                else:
                    if order.pickup_location not in route:
                        # print('CONSTRAINT FAIL ID 5')
                        return False, 5
                if order.delivery_location not in route:
                    # print('CONSTRAINT FAIL ID 6')
                    return False, 6
        else:
            # print('CONSTRAINT FAIL ID 7')
            return False, 7
        return True, 0

    def environment_validation(self):
        # for driver apply driver constraint vali
        for driver in self.env.drivers:
            validation, ids = self.driver_constraint_validation(driver=driver)
            if not validation:
                # print('DRIVER FAILS', driver.route, driver.inventory, driver.assigned)
                raise Exception(f'DRIVER {driver.id} FAILED VALIDATION ON {ids}')

        # for order in active check if is assigned exactly once
        active = self.env.active_orders.copy()
        for driver in self.env.drivers:
            for order in driver.assigned:
                if order not in active:
                    raise Exception(f'ORDER {order} FAILED ASSIGNMENT VALIDATION')
                active.remove(order)
        if active:
            raise Exception(f'UNASSIGNED ACTIVE ORDERS {active} FAILED VALIDATION')

    def solution_validation(self, solution, impacted):
        # for each impacted driver, validate route
        for driver in impacted:
            route = solution['routes'][driver]
            inventory = solution['inventories'][driver]
            assigned = solution['assigned'][driver]
            validation, _ = self.driver_constraint_validation(route=route, inventory=inventory, assigned=assigned)
            if not validation:
                return False
        return True

    def route_fitness(self, route, t, pareto=False):
        self.env.num_evaluations += 1
        # print('GET ROUTE FITNESS')
        # print(route)
        driver_costs = 0
        customer_costs = 0
        local_t = t
        for x in range(len(route)-1):
            loc = route[x]
            next_loc = route[x + 1]
            costs = abs(next_loc.x - loc.x) + abs(next_loc.y - loc.y)
            driver_costs += costs
            arrival = local_t + round(costs / 3)
            if next_loc.type == 'PICKUP':
                local_t = max(next_loc.order.apt, arrival)
            else:
                local_t = arrival
                customer_costs += (local_t - next_loc.order.apt)
            local_t += 2 # add service time
        if pareto:
            return [driver_costs, customer_costs]
        total_costs = self.alpha * driver_costs + self.beta * customer_costs
        # print('RETURN ROUTE FITNESS', total_costs)
        return total_costs

    def solution_fitness(self, solution, t, pareto=False):
    #     self.env.num_evaluations += 1
        # print('GET SOL FITNESS')
        if pareto:
            fitness = [0, 0]
        else:
            fitness = 0
        for route in solution['routes'].values():
            costs = self.route_fitness(route, t, pareto=pareto)
            if pareto:
                fitness[0] += costs[0]
                fitness[1] += costs[1]
            else:
                fitness += costs

        return fitness

    def delta_insert_cost(self, route, order, p, d, t):
        fitness = self.route_fitness(route, t)

        new_route = route.copy()
        new_route.insert(p, order.pickup_location)
        new_route.insert(d, order.delivery_location)

        new_fitness = self.route_fitness(new_route, t)
        return new_fitness - fitness

    def capacity_validation(self, inventory, route, order, p, d):
        current_capacity = len(inventory)

        new_route = route.copy()
        new_route.insert(p, order.pickup_location)
        new_route.insert(d, order.delivery_location)
        for location in new_route:
            if location.type == 'PICKUP':
                current_capacity += 1
                if current_capacity > self.env.capacity_constraint:
                    return False
            if location.type == 'DELIVERY':
                current_capacity -= 1
        return True

    def simple_greedy_optimization(self, t):
        # print('RUN SIMPLE GREEDY')
        # print(self.env.unassigned)
        assigned = []
        for order in self.env.unassigned:
            options = []
            for driver in self.env.drivers:
                for p in range(1, len(driver.route)+1):
                    for d in range(p+1, len(driver.route)+2):

                        optional_route = driver.route.copy()
                        optional_route.insert(p, order.pickup_location)
                        optional_route.insert(d, order.delivery_location)
                        optional_inventory = [str(order) for order in driver.inventory]
                        optional_assigned = driver.assigned.copy()

                        validation, _ = self.driver_constraint_validation(route=optional_route, inventory=optional_inventory, assigned=optional_assigned)
                        if validation:
                            cost = self.delta_insert_cost(driver.route, order, p, d, t)
                            options.append({'order': order, 'driver': driver, 'p': p, 'd': d, 'cost': cost})
            best = min(options, key=lambda x: x['cost'])

            best['driver'].assigned.append(best['order'])
            best['driver'].route.insert(best['p'], best['order'].pickup_location)
            best['driver'].route.insert(best['d'], best['order'].delivery_location)
            assigned.append(best['order'])
        for order in assigned:
            self.env.unassigned.remove(order)

    def greedy_solution(self, t):
        # print('RUN GREEDY SOLUTION')
        solution = {'routes': {}, 'inventories': {}, 'assigned': {}}
        assigned = []
        for driver in self.env.drivers:
            solution['routes'][driver.id] = [driver.current_location]
            solution['inventories'][driver.id] = driver.inventory.copy()
            solution['assigned'][driver.id] = []

            for location in driver.route:
                if location.type == 'DELIVERY':
                    if location.order in solution['inventories'][driver.id]:
                        solution['routes'][driver.id].append(location)
                        assigned.append(location.order)
                        solution['assigned'][driver.id].append(location.order)

        unassigned = []
        for order in self.env.active_orders:
            if order not in assigned:
                unassigned.append(order)

        while len(unassigned) > 0:
            options = []
            for driver in self.env.drivers:
                route = solution['routes'][driver.id]
                for order in unassigned:
                    for p in range(1, len(route)+1):
                        for d in range(p+1, len(route)+2):
                            if self.capacity_validation(solution['inventories'][driver.id], route, order, p, d):
                                cost = self.delta_insert_cost(route, order, p, d, t)
                                options.append({'order': order, 'driver': driver, 'p': p, 'd': d, 'cost': cost})
            min_costs = min(options, key=lambda x: x['cost'])
            max_costs = max(options, key=lambda x: x['cost'])
            threshold = min_costs['cost'] + 0.1 * (max_costs['cost'] - min_costs['cost'])
            rcl = [option for option in options if option['cost'] <= threshold]
            change = random.choice(rcl)
            solution['routes'][change['driver'].id].insert(change['p'], change['order'].pickup_location)
            solution['routes'][change['driver'].id].insert(change['d'], change['order'].delivery_location)
            solution['assigned'][change['driver'].id].append(change['order'])
            unassigned.remove(change['order'])

        return solution

    def copy_solution(self, solution):

        copied_solution = {
            'routes': {d_id: list(route) for d_id, route in solution['routes'].items()},
            'inventories': {d_id: list(inv) for d_id, inv in solution['inventories'].items()},
            'assigned': {d_id: list(assign) for d_id, assign in solution['assigned'].items()}
        }

        return copied_solution

    def re_ordering(self, solution):
        # print('RE-ORDERING')

        driver = random.choice(list(solution['routes'].keys()))

        if len(solution['routes'][driver]) > 2:

            swap = random.randint(1, len(solution['routes'][driver])-2)

            location1 = solution['routes'][driver][swap]
            location2 = solution['routes'][driver][swap + 1]

            new_solution = self.copy_solution(solution)
            new_solution['routes'][driver][swap + 1] = location1
            new_solution['routes'][driver][swap] = location2

            ids = [location1.order.id, location2.order.id]
            ids.sort()
            return new_solution, [driver], ('RE-ORDER', driver, ids[0], ids[1])

        return solution, [], []

    def relocation(self, solution):
        # print('RELOCATION')
        possible_drivers = []

        for driver in list(solution['routes'].keys()):
            for order in solution['assigned'][driver]:
                if order not in solution['inventories'][driver]:
                    possible_drivers.append(driver)
                    break

        if possible_drivers:
            driver1 = random.choice(possible_drivers)
            driver2_options = [driver for driver in list(solution['routes'].keys()) if driver != driver1]
            driver2 = random.choice(driver2_options)

            possible_orders = []
            for order in solution['assigned'][driver1]:
                if order not in solution['inventories'][driver1]:
                    possible_orders.append(order)

            moving_order = random.choice(possible_orders)

            new_solution = self.copy_solution(solution)

            new_solution['routes'][driver1] = [stop for stop in new_solution['routes'][driver1] if stop.order != moving_order]

            p_idx = random.randint(1, len(new_solution['routes'][driver2]))
            d_idx = random.randint(p_idx + 1, len(new_solution['routes'][driver2])+1)

            new_solution['routes'][driver2].insert(p_idx, moving_order.pickup_location)
            new_solution['routes'][driver2].insert(d_idx, moving_order.delivery_location)

            new_solution['assigned'][driver1].remove(moving_order)
            new_solution['assigned'][driver2].append(moving_order)

            ids = [driver1, driver2]
            ids.sort()

            return new_solution, [driver1, driver2], ('RELOCATE', moving_order.id, ids[0], ids[1])

        return solution, [], []

        # chose a driver with relocatable order
        # chose destination driver
        # relocate locations to new driver

        # return solution, impacted

    def swapping(self, solution):

        possible_drivers = []
        # print('SWAPPING', solution['routes'])

        for driver in list(solution['routes'].keys()):
            for order in solution['assigned'][driver]:
                if order not in solution['inventories'][driver]:
                    possible_drivers.append(driver)
                    break

        if len(possible_drivers) > 1:
            drivers = random.sample(possible_drivers, 2)

            possible_orders1 = []
            for order in solution['assigned'][drivers[0]]:
                if order not in solution['inventories'][drivers[0]]:
                    possible_orders1.append(order)

            possible_orders2 = []
            for order in solution['assigned'][drivers[1]]:
                if order not in solution['inventories'][drivers[1]]:
                    possible_orders2.append(order)

            swap_order1 = random.choice(possible_orders1)
            swap_order2 = random.choice(possible_orders2)

            new_solution = self.copy_solution(solution)

            new_solution['routes'][drivers[0]] = []
            new_solution['routes'][drivers[1]] = []

            for stop in solution['routes'][drivers[0]]:
                if stop.order == swap_order1:
                    if stop.type == 'PICKUP':
                        new_solution['routes'][drivers[0]].append(swap_order2.pickup_location)
                    else:
                        new_solution['routes'][drivers[0]].append(swap_order2.delivery_location)
                else:
                    new_solution['routes'][drivers[0]].append(stop)

            for stop in solution['routes'][drivers[1]]:
                if stop.order == swap_order2:
                    if stop.type == 'PICKUP':
                        new_solution['routes'][drivers[1]].append(swap_order1.pickup_location)
                    else:
                        new_solution['routes'][drivers[1]].append(swap_order1.delivery_location)
                else:
                    new_solution['routes'][drivers[1]].append(stop)


            new_solution['assigned'][drivers[0]].remove(swap_order1)
            new_solution['assigned'][drivers[0]].append(swap_order2)
            new_solution['assigned'][drivers[1]].remove(swap_order2)
            new_solution['assigned'][drivers[1]].append(swap_order1)

            # print('NEW SOLUTION', drivers, solution['routes'], new_solution['routes'])
            ids = [swap_order1.id, swap_order2.id]
            ids.sort()
            drivers.sort()
            return new_solution, drivers, ('SWAP', ids[0], ids[1], drivers[0], drivers[1])

        return solution, [], []

    def two_opt_star(self, solution):
        # print('TWO OPT')

        possible_drivers = []

        for driver in list(solution['routes'].keys()):
            for order in solution['assigned'][driver]:
                if order not in solution['inventories'][driver]:
                    possible_drivers.append(driver)
                    break

        if len(possible_drivers) > 1:
            driver1, driver2 = random.sample(possible_drivers, 2)

            cutoff1 = random.randint(1, len(solution['routes'][driver1]) - 2) # needs at least 2 for pickup and delivery location
            cutoff2 = random.randint(1, len(solution['routes'][driver2]) - 2)

            tail1_orders = [stop.order for stop in solution['routes'][driver1][cutoff1:] if stop.type == 'PICKUP']
            tail2_orders = [stop.order for stop in solution['routes'][driver2][cutoff2:] if stop.type == 'PICKUP']

            tail1 = [stop for stop in solution['routes'][driver1][cutoff1:] if stop.order in tail1_orders]
            tail2 = [stop for stop in solution['routes'][driver2][cutoff2:] if stop.order in tail2_orders]

            if not tail1 and not tail2:
                return solution, [], []

            new_solution = self.copy_solution(solution)

            new_solution['routes'][driver1] = [stop for stop in solution['routes'][driver1] if stop.order not in tail1_orders] + tail2
            new_solution['routes'][driver2] = [stop for stop in solution['routes'][driver2] if stop.order not in tail2_orders] + tail1

            for order in tail1_orders:
                new_solution['assigned'][driver1].remove(order)
                new_solution['assigned'][driver2].append(order)
            for order in tail2_orders:
                new_solution['assigned'][driver2].remove(order)
                new_solution['assigned'][driver1].append(order)

            ids = [[driver1, cutoff1], [driver2, cutoff2]]
            ids.sort(key=lambda x: x[1])

            return new_solution, [driver1, driver2], ('2-OPT*', ids[0][0], ids[1][0], ids[0][1], ids[1][1])

        return solution, [], []

    def local_search_mutation(self, solution):
        mutations = [self.re_ordering, self.relocation, self.swapping, self.two_opt_star]
        # print('APPLY LOCAL SEARCH')
        for _ in range(50): # to avoid infinite loop
            mutation = random.choice(mutations)
            for _ in range(5):
                new_solution, impacted, _ = mutation(solution)
                if impacted:
                    if self.solution_validation(new_solution, impacted):
                        # print('ACCEPTABLE NEW SOLUTION')
                        return new_solution
        return solution # if no valid mutation is found

    def repair_crossover(self, solution, t):
        assigned = []
        doubles = []
        for driver in list(solution['assigned'].keys()):
            for order in solution['assigned'][driver]:
                if order not in assigned:
                    assigned.append(order)
                else:
                    doubles.append(order)

        for order in doubles:
            drivers = []
            for driver in list(solution['assigned'].keys()):
                if order in solution['assigned'][driver]:
                    drivers.append(driver)
            driver = random.choice(drivers)
            solution['assigned'][driver].remove(order)

            solution['routes'][driver] = [loc for loc in solution['routes'][driver] if loc.order != order]

        unassigned = []
        for order in self.env.active_orders:
            if order not in assigned:
                unassigned.append(order)
        solution = self.greedy_insertion(t, solution, unassigned)

        return solution

    def crossover(self, parent1, parent2, t):

        num_cross = random.randint(1, round(self.num_drivers / 2))
        crossovers = random.sample(range(self.num_drivers), num_cross)

        child1 = self.copy_solution(parent1)
        child2 = self.copy_solution(parent2)

        for idx in crossovers:
            child1['routes'][idx] = parent2['routes'][idx].copy()
            child2['routes'][idx] = parent1['routes'][idx].copy()
            child1['assigned'][idx] = parent2['assigned'][idx].copy()
            child2['assigned'][idx] = parent1['assigned'][idx].copy()

        child1 = self.repair_crossover(child1, t)
        child2 = self.repair_crossover(child2, t)

        return child1, child2

    def get_pareto_winner(self, sample):
        best_candidate = sample[0]
        min_dominated_count = float('inf')

        for c1 in sample:
            dominated_count = 0
            for c2 in sample:
                if c1 is not c2 and self.dominates(c2[1], c1[1]):
                    dominated_count += 1

            if dominated_count < min_dominated_count:
                min_dominated_count = dominated_count
                best_candidate = c1

            if min_dominated_count == 0:
                break

        return best_candidate

    def tournament(self, population, tournament_size=3, pareto=False):
        # print('TOURNAMENT POP', population)
        candidates = random.sample(population, tournament_size)

        if pareto:
            winner1 = self.get_pareto_winner(candidates)
        else:
            winner1 = min(candidates, key=lambda x: x[1])

        possible_candidates = [c for c in population if c is not winner1]
        candidates = random.sample(possible_candidates, tournament_size)

        if pareto:
            winner2 = self.get_pareto_winner(candidates)
        else:
            winner2 = min(candidates, key=lambda x: x[1])

        return winner1[0], winner2[0]


    def create_neighborhood(self, t, solution, n_size=50):
        neighborhood = []
        mutations = [self.re_ordering, self.relocation, self.swapping, self.two_opt_star]
        for _ in range(n_size):
            mutation = random.choice(mutations)
            for _ in range(5):
                new_solution, impacted, move = mutation(solution)
                if impacted:
                    if self.solution_validation(new_solution, impacted):
                        fitness = self.solution_fitness(new_solution, t)
                        neighborhood.append({'solution': new_solution, 'move': move, 'fitness': fitness})

        return neighborhood

    def delta_remove_savings(self, solution, order, d_id, t):

        fitness = self.solution_fitness(solution, t)
        new_solution = self.copy_solution(solution)
        new_solution['assigned'][d_id].remove(order)
        new_solution['routes'][d_id] = [loc for loc in new_solution['routes'][d_id] if loc.order != order]
        new_fitness = self.solution_fitness(new_solution, t)

        savings = fitness - new_fitness

        return savings

    def random_removal(self, solution, t, n_remove=3):
        all_assigned = []
        for d_id, orders in solution['assigned'].items():
            all_assigned.extend([(d_id, o) for o in orders if o not in solution['inventories'][d_id]])

        to_remove = random.sample(all_assigned, min(n_remove, len(all_assigned)))
        for d_id, order in to_remove:
            solution['assigned'][d_id].remove(order)
            solution['routes'][d_id] = [location for location in solution['routes'][d_id] if location.order != order]
        removed = [o[1] for o in to_remove]
        return solution, removed

    def worst_removal(self, solution, t, n_remove=3):
        removal_costs = []
        for d_id, orders in solution['assigned'].items():
            for order in orders:
                if order not in solution['inventories'][d_id]:
                    cost_savings = self.delta_remove_savings(solution, order, d_id, t)
                    removal_costs.append((cost_savings, d_id, order))

        to_remove = []
        removal_costs.sort(key=lambda x: x[0], reverse=True)
        for i in range(min(n_remove, len(removal_costs))):
            _, d_id, order = removal_costs[i]
            solution['assigned'][d_id].remove(order)
            solution['routes'][d_id] = [loc for loc in solution['routes'][d_id] if loc.order != order]
            to_remove.append(order)
        return solution, to_remove

    def shaw_removal(self, solution, t, n_remove=3):
        all_orders = [o for d_id, orders in solution['assigned'].items() for o in orders if o not in solution['inventories'][d_id]]
        if not all_orders: return solution, []
        seed = random.choice(all_orders)

        related_scores = []
        for order in all_orders:
            dist_p = abs(seed.pickup_location.x - order.pickup_location.x) + abs(seed.pickup_location.y - order.pickup_location.y)
            dist_d = abs(seed.delivery_location.x - order.delivery_location.x) + abs(seed.delivery_location.y - order.delivery_location.y)
            time_diff = abs(seed.ept - order.ept)
            score = self.alpha * (dist_p + dist_d) + self.beta * time_diff
            related_scores.append((order, score))

        to_remove = []
        related_scores.sort(key=lambda x: x[1])
        for i in range(min(n_remove, len(related_scores))):
            order = related_scores[i][0]
            for d_id in solution['assigned']:
                if order in solution['assigned'][d_id]:
                    solution['assigned'][d_id].remove(order)
                    solution['routes'][d_id] = [loc for loc in solution['routes'][d_id] if loc.order != order]
            to_remove.append(order)
        return solution, to_remove

    def route_removal(self, solution, t):
        d_id = random.choice(list(solution['routes'].keys()))
        to_remove = [order for order in solution['assigned'][d_id] if order not in solution['inventories'][d_id]]
        solution['routes'][d_id] = [loc for loc in solution['routes'][d_id] if loc.order in solution['inventories'][d_id] or loc.type == 'DRIVER']
        solution['assigned'][d_id] = [order for order in solution['inventories'][d_id]]
        return solution, to_remove

    def greedy_repair(self, solution, t, removed):
        solution = self.greedy_insertion(t, solution, unassigned=removed, noise=False)
        return solution

    def noise_repair(self, solution, t, removed):
        solution = self.greedy_insertion(t, solution, unassigned=removed, noise=True)
        return solution

    def regret_n_repair(self, solution, t, removed, n=2):
        while removed:
            order_regrets = []
            for order in removed:
                costs = []
                for driver in self.env.drivers:
                    route = solution['routes'][driver.id]
                    for p in range(1, len(route) + 1):
                        for d in range(p + 1, len(route) + 2):
                            if self.capacity_validation(solution['inventories'][driver.id], route, order, p, d):
                                cost = self.delta_insert_cost(solution['routes'][driver.id], order, p, d, t)
                                costs.append(cost)

                costs.sort()
                regret_val = 0
                if len(costs) >= n:
                    for i in range(1, n):
                        regret_val += (costs[i] - costs[0])
                else:
                    regret_val = costs[0] if costs else 0

                order_regrets.append((regret_val, order))

            order_regrets.sort(key=lambda x: x[0], reverse=True)
            best_order = order_regrets[0][1]

            solution = self.greedy_insertion(t, solution, [best_order])
            removed.remove(best_order)

        return solution

    def regret_2_repair(self, solution, t, removed):
        solution = self.regret_n_repair(solution, t, removed, n=2)
        return solution

    def regret_3_repair(self, solution, t, removed):
        solution = self.regret_n_repair(solution, t, removed, n=3)
        return solution

    def generate_random_solution(self):
        solution = {'routes': {}, 'inventories': {}, 'assigned': {}}

        unassigned = self.env.active_orders.copy()

        for driver in self.env.drivers:
            solution['routes'][driver.id] = [driver.current_location]
            solution['inventories'][driver.id] = driver.inventory.copy()
            solution['assigned'][driver.id] = driver.inventory.copy()
            for order in solution['assigned'][driver.id]:
                unassigned.remove(order)

        # distribute unassigned orders randomly over drivers
        for order in unassigned:
            driver = random.choice(list(solution['assigned'].keys()))
            solution['assigned'][driver].append(order)

        # for each driver generate a route for its assigned orders
        for driver in list(solution['routes'].keys()):
            sim_inventory = solution['inventories'][driver].copy()
            remaining = solution['assigned'][driver].copy()

            while len(remaining) > 0:
                # if not full capacity randomly add a pickup from a new order or a delivery from an already planned order
                if len(sim_inventory) < self.env.capacity_constraint:
                    order = random.choice(remaining)
                    if order in sim_inventory: # add delivery is pickup has happened before otherwise add pickup
                        solution['routes'][driver].append(order.delivery_location)
                        remaining.remove(order)
                        sim_inventory.remove(order)
                    else:
                        solution['routes'][driver].append(order.pickup_location)
                        sim_inventory.append(order)
                else: # if capacity is full add delivery for one of these orders
                    order = random.choice(sim_inventory)
                    solution['routes'][driver].append(order.delivery_location)
                    remaining.remove(order)
                    sim_inventory.remove(order)

        # print('RETURN RANDOM SOLUTION', solution)

        return solution

    def greedy_randomized_search(self, t, max_iterations=100):
        # print('Greedy Randomized Search')
        best_solution = None
        best_fitness = np.inf

        for _ in range(max_iterations):
            solution = self.greedy_solution(t)
            local_solution = self.local_search_mutation(solution)
            local_fitness = self.solution_fitness(local_solution, t)
            if local_fitness < best_fitness:
                best_solution = local_solution
                best_fitness = local_fitness

        # apply best solution
        for driver in self.env.drivers:
            driver.route = best_solution['routes'][driver.id]
            driver.assigned = best_solution['assigned'][driver.id]
        self.env.unassigned = []

    def greedy_insertion(self, t, solution, unassigned=None, noise=False):
        if unassigned is None:
            unassigned = self.env.unassigned

        for order in unassigned:
            options = []
            for driver in self.env.drivers:
                route = solution['routes'][driver.id]
                for p in range(1, len(route)+1):
                    for d in range(p+1, len(route)+2):
                        if self.capacity_validation(solution['inventories'][driver.id], route, order, p, d):
                            cost = self.delta_insert_cost(solution['routes'][driver.id], order, p, d, t)
                            if noise:
                                cost += random.uniform(-5, 5)
                            options.append({'driver': driver, 'p': p, 'd': d, 'cost': cost})
            best = min(options, key=lambda x: x['cost'])

            solution['routes'][best['driver'].id].insert(best['p'], order.pickup_location)
            solution['routes'][best['driver'].id].insert(best['d'], order.delivery_location)
            solution['assigned'][best['driver'].id].append(order)
        return solution

    def simulated_annealing(self, t, T=100, T_min=0.1, alpha=0.95, k=50, max_stagnation=10):
        # print('Simulated Annealing')
        current_state = {'routes': {}, 'inventories': {}, 'assigned': {}}

        for driver in self.env.drivers:
            current_state['routes'][driver.id] = driver.route.copy()
            current_state['inventories'][driver.id] = driver.inventory.copy()
            current_state['assigned'][driver.id] = driver.assigned.copy()

        solution = self.greedy_insertion(t, current_state)
        best_solution = solution
        fitness = self.solution_fitness(solution, t)
        best_fitness = fitness
        stagnation = 0
        while T > T_min and stagnation < max_stagnation:
            found_improvement = False
            for _ in range(k):
                new_solution = self.local_search_mutation(solution)
                new_fitness = self.solution_fitness(new_solution, t)
                delta_costs = new_fitness - fitness
                if delta_costs < 0:
                    solution = new_solution
                    fitness = new_fitness
                    if new_fitness < best_fitness:
                        # print('ACCEPTING + NEW BEST', new_fitness)
                        best_solution = new_solution
                        best_fitness = new_fitness
                        stagnation = 0
                        found_improvement = True
                elif random.random() < np.exp(-delta_costs / T):
                    solution = new_solution
                    fitness = new_fitness


            if not found_improvement:
                stagnation += 1

            T = T * alpha

        # apply best solution (or current if no new best is found)
        for driver in self.env.drivers:
            driver.route = best_solution['routes'][driver.id]
            driver.assigned = best_solution['assigned'][driver.id]

        self.env.unassigned = []

    def dominates(self, fitness_A, fitness_B):
        return (fitness_A[0] <= fitness_B[0] and fitness_A[1] <= fitness_B[1]) and (fitness_A[0] < fitness_B[0] or fitness_A[1] < fitness_B[1])

    def genetic_algorithm(self, t, p_size=20, max_generations=100, mutation_rate=0.2, pareto=False):
        # print('Genetic Algorithm')

        current_state = {'routes': {}, 'inventories': {}, 'assigned': {}}

        for driver in self.env.drivers:
            current_state['routes'][driver.id] = driver.route.copy()
            current_state['inventories'][driver.id] = driver.inventory.copy()
            current_state['assigned'][driver.id] = driver.assigned.copy()

        solution = self.greedy_insertion(t, current_state)
        fitness = self.solution_fitness(solution, t, pareto=pareto)

        best_solution = solution
        best_fitness = fitness
        population = [[solution, fitness]]
        # print('STARTING BEST', best_fitness, best_solution)
        total_pop = []
        for _ in range(p_size - 1):
            solution = self.generate_random_solution()
            fitness = self.solution_fitness(solution, t, pareto=pareto)
            population.append([solution, fitness])
            if not pareto:
                if fitness < best_fitness:
                    best_solution = solution
                    best_fitness = fitness
                    # print('NEW BEST', best_fitness)
            else:
                total_pop.append(fitness)

        pareto_archive = []
        for _ in range(max_generations):
            new_population = []
            while len(new_population) < p_size:
                parent1, parent2 = self.tournament(population, pareto=pareto)
                child1, child2 = self.crossover(parent1, parent2, t)

                if random.random() < mutation_rate:
                    child1 = self.local_search_mutation(child1)
                if random.random() < mutation_rate:
                    child2 = self.local_search_mutation(child2)

                fitness1 = self.solution_fitness(child1, t, pareto=pareto)
                fitness2 = self.solution_fitness(child2, t, pareto=pareto)

                if not pareto:
                    if fitness1 < best_fitness:
                        best_solution = child1
                        best_fitness = fitness1
                    if fitness2 < best_fitness:
                        best_solution = child2
                        best_fitness = fitness2
                else:
                    total_pop.append(fitness1)
                    total_pop.append(fitness2)

                new_population.append([child1, fitness1])
                new_population.append([child2, fitness2])
            population = new_population

            if pareto:
                for solution, fitness in population:
                    dominated = any(self.dominates(par_fit, fitness) for par_sol, par_fit in pareto_archive)
                    if not dominated:
                        pareto_archive = [item for item in pareto_archive if not self.dominates(fitness, item[1])]
                        if [solution, fitness] not in pareto_archive:
                            pareto_archive.append([solution, fitness])
        if pareto:
            all_driver_costs = [item[0] for item in total_pop]
            all_customer_costs = [item[1] for item in total_pop]

            pareto_driver_costs = [item[1][0] for item in pareto_archive]
            pareto_customer_costs = [item[1][1] for item in pareto_archive]

            sorted_pareto = sorted(zip(pareto_driver_costs, pareto_customer_costs))
            sorted_dc, sorted_cc = zip(*sorted_pareto)
            print(f"t={t}, DC={pareto_driver_costs}, CC={pareto_customer_costs}")

            plt.figure(figsize=(10, 6))
            plt.scatter(all_driver_costs, all_customer_costs,
                        color='blue', alpha=0.3, s=25, label="Dominated Solutions (Sub-optimal)")
            plt.plot(sorted_dc, sorted_cc,
                     color='purple', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
            plt.scatter(pareto_driver_costs, pareto_customer_costs,
                        color='purple', s=60, zorder=3, label="Pareto-Optimal Front")

            plt.xlabel("Driving Distance", fontsize=11)
            plt.ylabel("Customer Waiting Times", fontsize=11)
            plt.title(f"Pareto Front vs. Discarded Solutions (at t={t})", fontsize=13, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none')

            plt.tight_layout()
            plt.show()
        else:
            # print('APPLY BEST SOLUTION', best_fitness, best_solution)
            # apply best solution (or current if no new best is found)
            for driver in self.env.drivers:
                driver.route = best_solution['routes'][driver.id]
                driver.assigned = best_solution['assigned'][driver.id]

            self.env.unassigned = []

    def artificial_bee_colony(self, t, p_size=20, iterations=100, l_max=10):
        # print('Artificial Bee Colony')

        current_state = {'routes': {}, 'inventories': {}, 'assigned': {}}

        for driver in self.env.drivers:
            current_state['routes'][driver.id] = driver.route.copy()
            current_state['inventories'][driver.id] = driver.inventory.copy()
            current_state['assigned'][driver.id] = driver.assigned.copy()

        source = self.greedy_insertion(t, current_state)
        fitness = self.solution_fitness(source, t)

        best_source = source
        best_fitness = fitness
        food_sources = [[source, fitness, 0]]

        for _ in range(p_size - 1):
            source = self.generate_random_solution()
            fitness = self.solution_fitness(source, t)
            food_sources.append([source, fitness, 0])
            if fitness < best_fitness:
                best_source = source
                best_fitness = fitness
                # print('NEW BEST', best_fitness)

        for _ in range(iterations):
            # employed bees
            for source_id in range(p_size):
                new_source = self.local_search_mutation(food_sources[source_id][0])
                new_fitness = self.solution_fitness(new_source, t)
                if new_fitness < food_sources[source_id][1]:
                    food_sources[source_id] = [new_source, new_fitness, 0]
                    if new_fitness < best_fitness:
                        best_source = new_source
                        best_fitness = new_fitness
                else:
                    food_sources[source_id][2] += 1

            # onlooker bees
            total_inv_fitness = sum(1 / (food_sources[source_id][1] + 1e-6) for source_id in range(p_size))
            weights = [(1 / (food_sources[source_id][1] + 1e-6)) / total_inv_fitness for source_id in range(p_size)]

            for _ in range(p_size):
                source_id = random.choices(range(p_size), weights=weights, k=1)[0]
                source = food_sources[source_id][0]
                new_source = self.local_search_mutation(source)
                new_fitness = self.solution_fitness(new_source, t)

                if new_fitness < food_sources[source_id][1]:
                    food_sources[source_id] = [new_source, new_fitness, 0]
                    if new_fitness < best_fitness:
                        best_source = new_source
                        best_fitness = new_fitness
                else:
                    food_sources[source_id][2] += 1

            # scout bees
            for source_id in range(p_size):
                if food_sources[source_id][2] >= l_max:
                    new_source = self.generate_random_solution()
                    new_fitness = self.solution_fitness(new_source, t)
                    food_sources[source_id] = [new_source, new_fitness, 0]
                    if new_fitness < best_fitness:
                        best_source = new_source
                        best_fitness = new_fitness

        # apply best source (or current if no new best is found)
        for driver in self.env.drivers:
            driver.route = best_source['routes'][driver.id]
            driver.assigned = best_source['assigned'][driver.id]

        self.env.unassigned = []

    def tabu_search(self, t, iterations=100, tabu_size=10, n_size=50):
        current_state = {'routes': {}, 'inventories': {}, 'assigned': {}}

        for driver in self.env.drivers:
            current_state['routes'][driver.id] = driver.route.copy()
            current_state['inventories'][driver.id] = driver.inventory.copy()
            current_state['assigned'][driver.id] = driver.assigned.copy()

        solution = self.greedy_insertion(t, current_state)
        fitness = self.solution_fitness(solution, t)

        best_solution = solution
        best_fitness = fitness

        tabu_list = []

        for _ in range(iterations):
            neighborhood = self.create_neighborhood(t, solution, n_size=n_size)

            if not neighborhood:
                break


            # sort neighborhood
            neighborhood.sort(key=lambda x: x['fitness'], reverse=True)

            for neighbor in neighborhood:
                if neighbor['move'] not in tabu_list or neighbor['fitness'] < best_fitness:
                    solution = neighbor['solution']
                    fitness = neighbor['fitness']

                    if fitness < best_fitness:
                        best_solution = solution
                        best_fitness = fitness

                    tabu_list.append(neighbor['move'])
                    if len(tabu_list) > tabu_size:
                        tabu_list.pop(0)

                    break

        # apply best source (or current if no new best is found)
        for driver in self.env.drivers:
            driver.route = best_solution['routes'][driver.id]
            driver.assigned = best_solution['assigned'][driver.id]

        self.env.unassigned = []


        # print('Tabu Search')
        # set current solution
        # set best solution
        # tabu_list = []
        # for _ in range(iterations)
            # create neighborhood
            # valid moves = []
            # for candidate in neighborhood
                # if candidate not in tabu_list
                    # add candidate to valid_moves
            # best move = min(fitness valid_moves)
            # if best_move < best_solution
                # update best_solution
            # update tabu list
            # solution = best move
        # return best_solution

    def adaptive_large_neighborhood_search(self, t, T=100, T_min=0.1, alpha=0.95, k=50, max_stagnation=10):
        # print('Adaptive Large Neighborhood Search')

        current_state = {'routes': {}, 'inventories': {}, 'assigned': {}}

        for driver in self.env.drivers:
            current_state['routes'][driver.id] = driver.route.copy()
            current_state['inventories'][driver.id] = driver.inventory.copy()
            current_state['assigned'][driver.id] = driver.assigned.copy()

        solution = self.greedy_insertion(t, current_state)
        fitness = self.solution_fitness(solution, t)

        best_solution = solution
        best_fitness = fitness

        destroy_methods = [self.random_removal, self.worst_removal, self.shaw_removal, self.route_removal]
        repair_methods = [self.greedy_repair, self.noise_repair, self.regret_2_repair, self.regret_3_repair]

        d_weights = np.ones(len(destroy_methods))
        r_weights = np.ones(len(repair_methods))

        stagnation = 0

        while T > T_min and stagnation < max_stagnation:
            found_improvement = False
            for _ in range(k):
                d_idx = random.choices(range(len(destroy_methods)), weights=d_weights, k=1)[0]
                r_idx = random.choices(range(len(repair_methods)), weights=r_weights, k=1)[0]

                destroy = destroy_methods[d_idx]
                repair = repair_methods[r_idx]

                copied_solution = self.copy_solution(solution)
                destroyed_solution, removed = destroy(copied_solution, t)
                new_solution = repair(destroyed_solution, t, removed)

                new_fitness = self.solution_fitness(new_solution, t)
                delta_costs = new_fitness - fitness

                if delta_costs < 0:
                    solution = new_solution
                    fitness = new_fitness
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
                        stagnation = 0
                        found_improvement = True
                        score = 10
                    else:
                        score = 5
                elif random.random() < np.exp(-delta_costs / T):
                    solution = new_solution
                    fitness = new_fitness
                    score = 2
                else:
                    score = 0

                d_weights[d_idx] = max(0.1, d_weights[d_idx] * 0.9 + score * 0.1)
                r_weights[r_idx] = max(0.1, r_weights[r_idx] * 0.9 + score * 0.1)

            if not found_improvement:
                stagnation += 1

            T *= alpha

        # apply best source (or current if no new best is found)
        for driver in self.env.drivers:
            driver.route = best_solution['routes'][driver.id]
            driver.assigned = best_solution['assigned'][driver.id]

        self.env.unassigned = []



        # set current solution
        # set best solution
        # initialize weights
        # while t > t min
            # select destroy operator
            # apply destroy
            # select repair operator
            # apply repair
            # if improving
                #
            # elif accepted
                #

    def update_env(self, t):
        for order in self.env.order_events:
            if t - 5 < order[1] <= t:
                if order[0].status != 'ACTIVE':
                    order[0].status = 'ACTIVE'
                    self.env.active_orders.append(order[0])
                    self.env.unassigned.append(order[0])

        for d in self.env.drivers:

            # print('BREAKDOWNS', self.breakdowns, d)
            if t in self.breakdowns[d.id]:
                d.broken = True
                # print('BREAKDOWN')
            if len(d.route) > 1:
                if str(d.current_location) == str(d.route[1]):
                    o = d.route[1].order
                    if t > o.ept:
                        o.current_cost += 1
                    if str(d.current_location) == str(o.pickup_location) and d.status != 'SERVICE':
                        if d.broken:
                            d.wait_time = 5
                            d.broken = False
                        if o.apt <= t:
                            d.inventory.append(o)
                            o.status = 'PICKED-UP'
                            d.wait_time += 2
                            d.status = 'SERVICE'
                            d.route.pop(1)
                        else:
                            d.status = 'WAITING'
                    if str(d.current_location) == str(o.delivery_location) and d.status != 'SERVICE':
                        d.inventory.remove(o)
                        d.assigned.remove(o)
                        self.env.active_orders.remove(o)
                        d.route.pop(1)
                        o.final_cost = t - o.ept
                        o.status = 'COMPLETED'
                        if d.broken:
                            d.wait_time = 7
                            d.broken = False
                        else:
                            d.wait_time = 2
                        d.status = 'SERVICE'

            d.wait_time = max(0, d.wait_time - 1)

            # update status if service time has passed
            if d.status == 'SERVICE' and d.wait_time == 0:
                d.status = 'IDLE'

            # update status if it can start driving
            if d.status == 'IDLE' and len(d.route) > 1:
                d.status = 'DRIVING'

            if d.status != 'SERVICE' and len(d.route) == 1:
                d.status = 'IDLE'

            if d.status == 'WAITING' and len(d.route) > 1:
                if str(d.current_location) != str(d.route[1]):
                    d.status = 'DRIVING'

    def move_drivers(self, t, speed=3):
        for driver in self.env.drivers:
            if len(driver.route) > 1:
                destination = driver.route[1]
            else:
                destination = Stop(round(self.env_size / 2), round(self.env_size / 2))
            move_x = 0
            move_y = 0
            if driver.status == 'DRIVING' or driver.status == 'IDLE':

                delta_x = destination.x - driver.current_location.x
                delta_y = destination.y - driver.current_location.y

                if driver.broken:
                    broken_speed = round(speed / 3)
                    speed = max(1, broken_speed)

                adjusted_speed = max(0, speed - self.traffic_delays[driver.id].count(t))
                # print('NEW SPEED', adjusted_speed)
                for _ in range(adjusted_speed):
                    if abs(delta_x - move_x) == 0 and abs(delta_y - move_y) == 0: # check if destination is reached
                        break
                    elif abs(delta_x - move_x) >= abs(delta_y - move_y):
                        move_x += 1 if delta_x > 0 else -1
                    else:
                        move_y += 1 if delta_y > 0 else -1

                new_x = driver.current_location.x + move_x
                new_y = driver.current_location.y + move_y

                driver.current_location.x = new_x
                driver.current_location.y = new_y
                driver.route[0] = driver.current_location

                driven_dist = abs(move_x) + abs(move_y)
                self.driver_costs += driven_dist

    # save as json file
    def save_logs(self, title):
        with open(f'logs/{title}.json', 'w') as f:
            json.dump(self.log, f, indent=4)

    # load from json files
    def load_logs(self, title):
        with open(f'{title}.json', 'r') as f:
            self.log = json.load(f)


    def update_logs(self, opt, t):
        active_orders = 0
        picked_orders = 0
        completed_orders = 0
        customer_costs = 0
        for order in self.env.order_events:
            if order[0].status == 'ACTIVE':
                active_orders += 1
                customer_costs += order[0].current_cost
            elif order[0].status == 'PICKED-UP':
                picked_orders += 1
                customer_costs += order[0].current_cost
            elif order[0].status == 'COMPLETED':
                completed_orders += 1
                customer_costs += order[0].final_cost

        routes = {}
        inventories = {}
        driver_states = {}
        for driver in self.env.drivers:
            inventories[driver.id] = len(driver.inventory)
            driver_states[driver.id] = driver.status
            route = []
            for location in driver.route:
                if location.type == 'DRIVER':
                    route.append((str(location), str(location.type), driver.status, str(driver.wait_time)))
                else:
                    route.append((str(location), str(location.type), str(location.order.apt)))
            routes[driver.id] = route

        total_costs = round(self.alpha * self.driver_costs + self.beta * customer_costs)

        new_log = {'ACTIVE': active_orders, 'PICKED-UP': picked_orders, 'COMPLETED': completed_orders, 'TC': total_costs, 'DC': self.driver_costs, 'CC': customer_costs, 'DRIVERS': driver_states, 'INVENTORIES': inventories, 'ROUTES': routes}

        # print('NEW LOG', new_log)
        self.log[self.run_x][opt][t] = new_log
        # print(f'T: {t} {new_log}')

    def run(self, repeats=10, optimizers=None, visualize=False, pareto=False):
        for i in range(repeats):
            self.run_x = i
            self.log[i] = {}

            self.order_events = self.initialize_orders(visualize)
            self.traffic_delays = self.initialize_traffic_delays()
            self.breakdowns = self.initialize_breakdowns()

            if optimizers is None:
                print('DEBUG RUN')
            else:
                for opt in optimizers:
                    self.env = Environment(num_drivers=self.num_drivers, env_size=self.env_size)
                    self.driver_costs = 0
                    self.customer_costs = 0
                    self.total_costs = 0
                    self.set_stochasticity()
                    match opt:
                        case 'GRASP':
                            optimizer = self.greedy_randomized_search
                        case 'SA':
                            optimizer = self.simulated_annealing
                        case 'GA':
                            optimizer = self.genetic_algorithm
                        case 'ABC':
                            optimizer = self.artificial_bee_colony
                        case 'TABU':
                            optimizer = self.tabu_search
                        case 'ALNS':
                            optimizer = self.adaptive_large_neighborhood_search
                        case _:
                            print('OPTIMIZER NOT RECOGNIZED')
                            optimizer = self.simple_greedy_optimization
                    print(f'START SIMULATION {opt} ON ENVIRONMENT {i}')
                    self.log[i][opt] = {}
                    pbar = tqdm(total=self.t_max, desc=f"Simulating {opt}", ncols=80, leave=True, mininterval=0.1, maxinterval=1.0, miniters=1)
                    timer0 = time.process_time()
                    t = 0
                    while t < self.t_max or len(self.env.active_orders) > 0:

                    # for t in tqdm(range(self.t_max)):
                        self.move_drivers(t)
                        self.update_env(t)

                        if t % 5 == 0:
                            if pareto:
                                self.genetic_algorithm(t, pareto=True)
                            optimizer(t)
                            self.environment_validation()

                        self.update_logs(opt, t)
                        t += 1
                        pbar.update(1)
                    timer1 = time.process_time()
                    self.log[i][opt]['evaluations'] = self.env.num_evaluations
                    self.log[i][opt]['time'] = timer1 - timer0
                    print()
                    pbar.close()
                    print('LOG:', self.log[i][opt][t -1])
                self.save_logs(f'simulation_results_{self.traffic_delay_rate}_{self.rest_delay_rate}_{self.breakdown_rate}_{self.env_size}_{self.num_orders}_{self.num_drivers}_{repeats}')

    def analysis(self, title=''):
        print('FINAL LOG', self.log)
        total_time = {}
        run_time = {}
        evaluations = {}
        fitness = {}
        driver_costs = {}
        customer_costs = {}
        total_batching = {}
        active_batching = {}
        for run in self.log.keys():
            for opt in self.log[run].keys():
                if opt in ['GRASP', 'SA', 'GA', 'ABC', 'TABU', 'ALNS']:
                    max_t = list(self.log[run][opt].keys())[-3]
                    if opt not in total_time:
                        total_time[opt] = []
                    total_time[opt].append(max_t)
                    if opt not in run_time:
                        run_time[opt] = []
                    run_time[opt].append(self.log[run][opt]['time'])
                    if opt not in evaluations:
                        evaluations[opt] = []
                    evaluations[opt].append(self.log[run][opt]['evaluations'])
                    if opt not in fitness:
                        fitness[opt] = []
                    fitness[opt].append(self.log[run][opt][max_t]['TC'])
                    if opt not in driver_costs:
                        driver_costs[opt] = []
                    driver_costs[opt].append(self.log[run][opt][max_t]['DC'])
                    if opt not in customer_costs:
                        customer_costs[opt] = []
                    customer_costs[opt].append(self.log[run][opt][max_t]['CC'])
                    if opt not in total_batching:
                        total_batching[opt] = []
                    carried = 0
                    for t in list(self.log[run][opt].keys())[:-2]:
                        inv = sum(list(self.log[run][opt][t]['INVENTORIES'].values()))
                        carried += inv
                    tbu = carried / (max_t * self.num_drivers) # total batching utilization
                    total_batching[opt].append(tbu)
                    if opt not in active_batching:
                        active_batching[opt] = []
                    active_inv = 0
                    active_t = 0
                    for t in list(self.log[run][opt].keys())[:-2]:
                        for d in range(self.num_drivers):
                            if len(self.log[run][opt][t]['ROUTES'][d]) > 1:
                                active_inv += self.log[run][opt][t]['INVENTORIES'][d]
                                active_t += 1
                    abu = active_inv / active_t
                    active_batching[opt].append(abu)

        print('TOTAL TIME', total_time)
        print('RUN TIME', run_time)
        print('EVALUATIONS', evaluations)
        print('FITNESS', fitness)
        print('DRIVER COSTS', driver_costs)
        print('CUSTOMER COSTS', customer_costs)
        print('TOTAL BATCHING', total_batching)
        print('ACTIVE BATCHING', active_batching)

        colors = ['#88CCEE', '#CC6677', '#DDCC77', '#117733', '#332288', '#AA4499']

        for dict, type_title, ylabel in [[total_time, 'Total Time', 'Timesteps'], [run_time, 'Run Time', 'Time (s)'],
                                         [evaluations, 'Evaluations', 'Num evaluations'],
                                         [fitness, 'Fitness', 'Fitness'],
                                         [driver_costs, 'Driver Costs', 'Traveled distance'],
                                         [customer_costs, 'Customer Costs', 'Waiting Time (m)'],
                                         [total_batching, 'Total Batching', 'Average inventory size'],
                                         [active_batching, 'Active Batching', 'Average inventory size']]:
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(list(dict.values()), tick_labels=list(dict.keys()), patch_artist=True)
            for box, color in zip(bp['boxes'], colors):
                box.set_facecolor(color)
                box.set_alpha(0.7)
            ax.set_title(f'{title} {type_title}')
            ax.set_ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(f'figures/{title}_{type_title}.png')
            plt.show()

