class Order:
    def __init__(self, id, pickup, delivery, status, ept, apt):
        self.id = id
        self.pickup_location = pickup
        self.delivery_location = delivery
        self.status = status
        self.ept = ept
        self.apt = apt
        self.final_cost = 0
        self.current_cost = 0

    def __repr__(self):
        return '(P:' + str(self.pickup_location) + '),(D:' + str(self.delivery_location) + ')'


class Driver:
    def __init__(self, id, current_x, current_y):
        self.id = id
        self.current_location = Stop(current_x, current_y, 'DRIVER')
        self.current_load = 0
        self.route = [self.current_location]
        self.broken = False
        self.assigned = []
        self.status = 'IDLE'
        self.inventory = []
        self.wait_time = 0

    def __repr__(self):
        return str(self.id)

class Stop:
    def __init__(self, x, y, type=None, order=None):
        self.x = x
        self.y = y
        self.type = type
        self.order = order

    def __repr__(self):
        return str(self.x) + ',' + str(self.y)

    def __str__(self):
        return str(self.x) + "," + str(self.y)


class Environment:
    def __init__(self, num_drivers=10, num_orders=100, capacity=5, env_size=50):
        self.order_events = []
        self.traffic_delays = []
        self.breakdowns = []
        self.active_orders = []
        self.drivers = []
        self.num_drivers = num_drivers
        self.num_orders = num_orders
        self.unassigned = []
        self.capacity_constraint = capacity
        self.num_evaluations = 0

        for i in range(num_drivers):
            self.drivers.append(Driver(id=i, current_x=round(env_size / 2), current_y=round(env_size / 2)))
