"""
Enterprise Supply Chain Environment - Multi-echelon Inventory Optimization with Stochastic Demand
"""

from __future__ import annotations
import gym
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Facility:
    """Supply chain network node definition"""
    id: str
    type: str  # supplier, factory, warehouse, retailer
    location: Tuple[float, float]  # (lat, lon)
    capacity: int
    lead_time: int  # days
    holding_cost: float  # $/unit/day
    operating_cost: float  # $/day

@dataclass  
class TransportationRoute:
    """Logistics route between facilities"""
    origin: str
    destination: str
    cost_per_unit: float  # $/unit
    transit_time: int  # days
    reliability: float  # 0-1 probability

class SupplyChainEnv(gym.Env):
    """Multi-echelon supply chain optimization environment"""
    
    metadata = {'render.modes': ['human', 'system']}
    
    def __init__(self, config: Dict):
        """
        Args:
            config: {
                "network": List[Facility],
                "routes": List[TransportationRoute],
                "demand_model": Dict,  # ARIMA parameters
                "max_steps": int,
                "initial_state": Dict,
                "reward_weights": {
                    "service_level": 1.0,
                    "inventory_cost": -0.01,
                    "transport_cost": -0.005,
                    "backorder_cost": -0.1
                }
            }
        """
        super().__init__()
        
        # Network configuration
        self.network = {f.id: f for f in config['network']}
        self.routes = config['routes']
        self.max_steps = config.get('max_steps', 365)
        
        # Demand forecasting model
        self.demand_model = DemandModel(**config['demand_model'])
        
        # Initialize state
        self.state = self._init_state(config.get('initial_state'))
        
        # Define action/observation spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        
        # Reward parameters  
        self.reward_weights = config['reward_weights']
        
        # History tracking
        self.history = {
            'inventory': [],
            'orders': [],
            'shipments': [],
            'demand': []
        }

    def _init_state(self, initial_state: Dict) -> Dict:
        """Initialize supply chain state"""
        default_state = {
            'inventory': {f.id: f.capacity//2 for f in self.network.values()},
            'in_transit': defaultdict(list),  # {route: [(qty, arrival_date)]}
            'backorders': defaultdict(int),
            'current_date': datetime.now(),
            'demand_forecast': self.demand_model.initial_forecast()
        }
        return {**default_state, **(initial_state or {})}

    def _build_action_space(self) -> gym.Space:
        """Define multi-dimensional action space"""
        return gym.spaces.Dict({
            f.id: gym.spaces.Box(
                low=0, 
                high=1, 
                shape=(len(self._get_valid_destinations(f.id)),)
            ) for f in self.network.values()
        })

    def _build_observation_space(self) -> gym.Space:
        """Define complex observation space"""
        return gym.spaces.Dict({
            'inventory': gym.spaces.Box(low=0, high=np.inf, shape=(len(self.network),)),
            'in_transit': gym.spaces.Sequence(
                gym.spaces.Box(low=0, high=np.inf, shape=(3,))  # (qty, days_remaining, route_id)
            ),
            'demand_forecast': gym.spaces.Box(low=0, high=np.inf, shape=(30,)),  # 30-day forecast
            'facility_utilization': gym.spaces.Box(low=0, high=1, shape=(len(self.network),))
        })

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute one supply chain time step"""
        
        # 1. Process incoming shipments
        self._update_inventory()
        
        # 2. Generate customer demand
        actual_demand = self._generate_demand()
        
        # 3. Fulfill orders
        fulfilled, backordered = self._fulfill_orders(actual_demand)
        
        # 4. Process agent actions (replenishment orders)
        self._process_actions(action)
        
        # 5. Advance time
        self.state['current_date'] += timedelta(days=1)
        
        # 6. Calculate reward
        reward = self._calculate_reward(fulfilled, backordered)
        
        # 7. Check termination
        done = self._steps >= self.max_steps
        
        # 8. Update history
        self._update_history(fulfilled, backordered)
        
        return self.state, reward, done, self._get_info(fulfilled, backordered)

    def _update_inventory(self):
        """Process incoming shipments and inventory aging"""
        for route in list(self.state['in_transit']):
            for shipment in list(self.state['in_transit'][route]):
                shipment['days_remaining'] -= 1
                if shipment['days_remaining'] <= 0:
                    self.state['inventory'][route.destination] += shipment['qty']
                    self.state['in_transit'][route].remove(shipment)

    def _generate_demand(self) -> Dict:
        """Generate stochastic demand across retail nodes"""
        return {
            node.id: self.demand_model.predict(
                self.state['current_date'],
                self.state['demand_forecast']
            ) for node in self.network.values() 
            if node.type == 'retailer'
        }

    def _fulfill_orders(self, demand: Dict) -> Tuple[Dict, Dict]:
        """Attempt to fulfill customer demand with current inventory"""
        fulfilled = {}
        backordered = {}
        
        for node_id, qty in demand.items():
            available = self.state['inventory'][node_id]
            fulfilled_qty = min(available, qty)
            
            fulfilled[node_id] = fulfilled_qty
            backordered[node_id] = qty - fulfilled_qty
            
            self.state['inventory'][node_id] -= fulfilled_qty
            self.state['backorders'][node_id] += backordered[node_id]
            
        return fulfilled, backordered

    def _process_actions(self, action: Dict):
        """Convert agent actions to replenishment orders"""
        for node_id, action_vector in action.items():
            valid_destinations = self._get_valid_destinations(node_id)
            total_order = np.sum(action_vector)
            
            if total_order > 0:
                allocated = self._allocate_order(node_id, action_vector, valid_destinations)
                self._place_orders(node_id, allocated)

    def _allocate_order(self, node_id: str, action: np.ndarray, destinations: List[str]) -> Dict:
        """Distribute order quantities across valid destinations"""
        normalized = action / (np.sum(action) + 1e-8)
        max_order = self.network[node_id].capacity - self.state['inventory'][node_id]
        return {
            dest: normalized[i] * max_order
            for i, dest in enumerate(destinations)
        }

    def _place_orders(self, origin: str, allocations: Dict):
        """Create shipments for each order"""
        for dest, qty in allocations.items():
            if qty > 0:
                route = self._get_route(origin, dest)
                transit_time = self._calculate_transit_time(route)
                
                self.state['in_transit'][route].append({
                    'qty': qty,
                    'days_remaining': transit_time,
                    'cost': qty * route.cost_per_unit
                })
                
                self.state['inventory'][origin] += qty

    def _calculate_transit_time(self, route: TransportationRoute) -> int:
        """Calculate actual transit time with reliability factors"""
        base_time = route.transit_time
        failure = np.random.random() > route.reliability
        return base_time + 2 if failure else base_time

    def _calculate_reward(self, fulfilled: Dict, backordered: Dict) -> float:
        """Compute multi-objective reward signal"""
        service_level = sum(fulfilled.values()) / (sum(fulfilled.values()) + sum(backordered.values()) + 1e-8)
        
        inventory_cost = sum(
            self.network[nid].holding_cost * inv 
            for nid, inv in self.state['inventory'].items()
        )
        
        transport_cost = sum(
            sum(ship['cost'] for ship in shipments)
            for shipments in self.state['in_transit'].values()
        )
        
        backorder_cost = sum(
            qty * self.network[nid].holding_cost * 2  # Penalize backorders more
            for nid, qty in self.state['backorders'].items()
        )
        
        return (
            self.reward_weights['service_level'] * service_level +
            self.reward_weights['inventory_cost'] * inventory_cost +
            self.reward_weights['transport_cost'] * transport_cost +
            self.reward_weights['backorder_cost'] * backorder_cost
        )

    def reset(self) -> Dict:
        """Reset environment to initial state"""
        self.__init__(self.config)
        return self.state

    def render(self, mode='human'):
        """Visualize supply chain state"""
        if mode == 'human':
            self._plot_network_status()
        elif mode == 'system':
            return self._generate_system_report()

    def _plot_network_status(self):
        """Visualization of inventory levels and shipments"""
        plt.figure(figsize=(15, 8))
        
        # Inventory levels
        plt.subplot(2, 2, 1)
        sns.barplot(
            x=list(self.state['inventory'].keys()),
            y=list(self.state['inventory'].values())
        )
        plt.title('Current Inventory Levels')
        
        # In-transit shipments
        plt.subplot(2, 2, 2)
        transit_qty = [len(r) for r in self.state['in_transit'].values()]
        sns.barplot(
            x=list(self.state['in_transit'].keys()),
            y=transit_qty
        )
        plt.title('In-Transit Shipments')
        
        # Demand vs Fulfillment
        plt.subplot(2, 2, 3)
        last_demand = self.history['demand'][-30:]
        sns.lineplot(data=pd.DataFrame(last_demand))
        plt.title('Recent Demand Pattern')
        
        plt.tight_layout()
        plt.show()

class DemandModel:
    """ARIMA-based demand forecasting model with seasonality"""
    
    def __init__(self, p: int, d: int, q: int, seasonal_period: int):
        self.p = p
        self.d = d 
        self.q = q
        self.seasonal_period = seasonal_period
        self.model = self._fit_initial_model()
        
    def _fit_initial_model(self):
        # Implementation placeholder for actual ARIMA model
        return None
        
    def predict(self, current_date: datetime, history: List[float]) -> float:
        """Generate stochastic demand prediction"""
        # Base demand + seasonality + noise
        base = 100  # Baseline daily demand
        seasonality = 50 * np.sin(2*np.pi*(current_date.dayofyear/365))
        noise = np.random.normal(0, 20)
        return max(0, base + seasonality + noise)

# Example Usage
if __name__ == "__main__":
    config = {
        "network": [
            Facility("supplier1", "supplier", (37.7749, -122.4194), 10000, 7, 0.1, 500),
            Facility("factory1", "factory", (34.0522, -118.2437), 5000, 3, 0.2, 1000),
            Facility("warehouse1", "warehouse", (40.7128, -74.0060), 8000, 2, 0.15, 800),
            Facility("retail1", "retailer", (41.8781, -87.6298), 2000, 1, 0.3, 300)
        ],
        "routes": [
            TransportationRoute("supplier1", "factory1", 0.5, 7, 0.95),
            TransportationRoute("factory1", "warehouse1", 0.3, 5, 0.9),
            TransportationRoute("warehouse1", "retail1", 0.2, 3, 0.85)
        ],
        "demand_model": {"p": 2, "d": 1, "q": 2, "seasonal_period": 365},
        "max_steps": 365,
        "reward_weights": {
            "service_level": 1.0,
            "inventory_cost": -0.01,
            "transport_cost": -0.005,
            "backorder_cost": -0.1
        }
    }
    
    env = SupplyChainEnv(config)
    obs = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        
        if done:
            break
