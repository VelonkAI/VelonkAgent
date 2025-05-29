"""
High-Fidelity Financial Market Environment - Multi-Asset Trading with Order Book Dynamics
"""

from __future__ import annotations
import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import multivariate_normal
from sortedcontainers import SortedDict

@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_type: str  # 'EQUITY', 'FUTURE', 'OPTION'
    tick_size: float
    lot_size: int
    fee_structure: Dict[str, float]  # {'maker': 0.0002, 'taker': 0.0005}

@dataclass
class MarketEvent:
    event_type: str  # 'CORRELATION_SHIFT', 'FLASH_CRASH', 'NEWS'
    affected_assets: List[str]
    parameters: Dict

class FinancialMarketEnv(gym.Env):
    """Institutional trading environment with multi-timescale order book dynamics"""
    
    metadata = {'render.modes': ['candlestick', 'order_book', 'portfolio']}
    
    def __init__(self, config: Dict):
        """
        Args:
            config: {
                "assets": List[Asset],
                "initial_balance": float,
                "position_limits": Dict[str, Tuple[float, float]],
                "data_source": str,  # 'synthetic' or 'historical'
                "data_params": Dict,
                "reward_components": {
                    "sharpe_ratio": 0.7,
                    "drawdown_penalty": -0.3,
                    "turnover_cost": -0.2
                },
                "market_event_prob": 0.05
            }
        """
        super().__init__()
        
        # Market configuration
        self.assets = {a.symbol: a for a in config['assets']}
        self.position_limits = config['position_limits']
        self.fee_schedule = self._build_fee_schedule()
        
        # Initialize state
        self.state = self._init_state(config)
        
        # Data generation system
        self.data_gen = self._init_data_system(config['data_source'], config['data_params'])
        
        # Define spaces
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        
        # Event system
        self.market_event_prob = config.get('market_event_prob', 0.05)
        self.active_events: List[MarketEvent] = []
        
        # History tracking
        self.history = {
            'portfolio_value': [],
            'positions': [],
            'market_events': []
        }

    def _init_state(self, config: Dict) -> Dict:
        """Initialize trading state"""
        return {
            'portfolio': {
                'cash': config['initial_balance'],
                'positions': {sym: 0.0 for sym in self.assets},
                'valuation': config['initial_balance']
            },
            'order_book': self._init_order_books(),
            'market_data': next(self.data_gen),
            'risk_metrics': {
                'var_95': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            },
            'step': 0
        }

    def _build_action_space(self) -> gym.Space:
        """Multi-dimensional action space for trading signals"""
        return gym.spaces.Dict({
            sym: gym.spaces.Box(
                low=np.array([-1, 0]),  # [direction (-1=short, 1=long), intensity]
                high=np.array([1, 1]),
                dtype=np.float32
            ) for sym in self.assets
        })

    def _build_observation_space(self) -> gym.Space:
        """Complex observation space with market state"""
        return gym.spaces.Dict({
            'order_book_depth': gym.spaces.Box(
                low=0, high=np.inf, 
                shape=(len(self.assets), 5, 2)  # (asset, level, bid/ask)
            ),
            'portfolio_state': gym.spaces.Dict({
                'cash': gym.spaces.Box(0, np.inf, (1,)),
                'positions': gym.spaces.Box(
                    low=np.array([self.position_limits[sym][0] for sym in self.assets]),
                    high=np.array([self.position_limits[sym][1] for sym in self.assets])
                ),
                'exposure': gym.spaces.Box(0, 1, (len(self.assets),))
            }),
            'market_features': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(len(self.assets), 10)  # OHLCV + technical indicators
            ),
            'risk_state': gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3,)  # VAR, drawdown, volatility
            )
        })

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute one trading interval"""
        
        # 1. Process market events
        self._generate_market_events()
        
        # 2. Update market data
        self.state['market_data'] = next(self.data_gen)
        
        # 3. Process agent actions
        self._execute_orders(action)
        
        # 4. Update portfolio valuation
        self._mark_to_market()
        
        # 5. Calculate reward
        reward = self._calculate_reward()
        
        # 6. Update risk metrics
        self._update_risk_metrics()
        
        # 7. Check termination
        done = self._check_termination()
        
        # 8. Record history
        self._update_history()
        
        return self.state, reward, done, self._get_info()

    def _execute_orders(self, action: Dict):
        """Convert trading signals to executable orders"""
        for sym, (direction, intensity) in action.items():
            if abs(intensity) < 0.1:  # Noise threshold
                continue
                
            current_pos = self.state['portfolio']['positions'][sym]
            price = self._best_mid_price(sym)
            fee_rate = self._determine_fee(sym, direction)
            
            # Calculate target position
            target = direction * intensity * self.position_limits[sym][1]
            delta = target - current_pos
            
            if delta != 0:
                # Simulate order execution with market impact
                executed_qty, executed_price = self._simulate_order_execution(
                    sym, delta, price
                )
                
                # Update portfolio
                cost = executed_qty * executed_price
                fee = abs(cost) * fee_rate
                
                self.state['portfolio']['positions'][sym] += executed_qty
                self.state['portfolio']['cash'] -= cost + fee
                
                # Update order book
                self._apply_market_impact(sym, executed_qty, executed_price)

    def _simulate_order_execution(self, sym: str, qty: float, ref_price: float) -> Tuple[float, float]:
        """Simulate realistic order execution with slippage"""
        asset = self.assets[sym]
        book = self.state['order_book'][sym]
        remaining = qty
        total_cost = 0.0
        filled_qty = 0.0
        
        if qty > 0:  # Buy order
            for price, volume in book['ask'].items():
                if remaining <= 0:
                    break
                fill = min(remaining, volume)
                filled_qty += fill
                total_cost += fill * price
                remaining -= fill
        else:  # Sell order
            for price, volume in book['bid'].items():
                if remaining >= 0:
                    break
                fill = max(remaining, -volume)
                filled_qty += fill
                total_cost += fill * price
                remaining -= fill
                
        # Apply price impact
        impact_factor = 0.0001 * abs(qty)/asset.lot_size
        impact = ref_price * impact_factor
        executed_price = (total_cost / filled_qty) if filled_qty !=0 else ref_price
        executed_price += impact if qty >0 else -impact
        
        return filled_qty, executed_price

    def _mark_to_market(self):
        """Revalue portfolio based on current market prices"""
        total = self.state['portfolio']['cash']
        for sym, qty in self.state['portfolio']['positions'].items():
            price = self._best_mid_price(sym)
            total += qty * price
        self.state['portfolio']['valuation'] = total

    def _calculate_reward(self) -> float:
        """Compute multi-factor reward signal"""
        returns = np.diff(self.history['portfolio_value'][-30:], prepend=self.history['portfolio_value'][0])
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
        
        current_drawdown = (max(self.history['portfolio_value']) - self.state['portfolio']['valuation']) / \
                           max(self.history['portfolio_value'])
        
        turnover = sum(
            abs(self.state['portfolio']['positions'][sym] - 
                self.history['positions'][-1][sym]) * self._best_mid_price(sym)
            for sym in self.assets
        ) / self.state['portfolio']['valuation']
        
        return (
            self.config['reward_components']['sharpe_ratio'] * sharpe +
            self.config['reward_components']['drawdown_penalty'] * current_drawdown +
            self.config['reward_components']['turnover_cost'] * turnover
        )

    def _update_risk_metrics(self):
        """Compute Value-at-Risk and other risk measures"""
        returns = pd.Series(np.diff(self.history['portfolio_value']))
        self.state['risk_metrics']['var_95'] = returns.quantile(0.05)
        self.state['risk_metrics']['volatility'] = returns.std()
        
        peak = max(self.history['portfolio_value'])
        trough = min(self.history['portfolio_value'][self.history['portfolio_value'].index(peak):])
        self.state['risk_metrics']['max_drawdown'] = (peak - trough) / peak

    def _generate_market_events(self):
        """Simulate market microstructure events"""
        if np.random.rand() < self.market_event_prob:
            event_type = np.random.choice([
                'CORRELATION_SHIFT', 
                'FLASH_CRASH', 
                'LIQUIDITY_SHOCK'
            ])
            
            affected = np.random.choice(
                list(self.assets.keys()), 
                size=np.random.randint(1, len(self.assets))
            )
            
            event = MarketEvent(
                event_type=event_type,
                affected_assets=affected,
                parameters=self._event_parameters(event_type)
            )
            
            self._apply_market_event(event)
            self.active_events.append(event)
            self.history['market_events'].append(event)

    def _event_parameters(self, event_type: str) -> Dict:
        """Generate parameters for market events"""
        params = {
            'CORRELATION_SHIFT': {
                'correlation_matrix': np.random.normal(0, 0.8, (len(self.assets), len(self.assets)))
            },
            'FLASH_CRASH': {
                'magnitude': np.random.uniform(-0.15, -0.05),
                'duration': np.random.randint(5, 30)
            },
            'LIQUIDITY_SHOCK': {
                'spread_increase': np.random.uniform(2.0, 5.0),
                'depth_reduction': np.random.uniform(0.5, 0.8)
            }
        }
        return params.get(event_type, {})

    def _apply_market_event(self, event: MarketEvent):
        """Modify market state based on events"""
        if event.event_type == 'FLASH_CRASH':
            for sym in event.affected_assets:
                mid = self._best_mid_price(sym)
                new_price = mid * (1 + event.parameters['magnitude'])
                self._adjust_order_book(sym, new_price)
                
        elif event.event_type == 'LIQUIDITY_SHOCK':
            for sym in event.affected_assets:
                book = self.state['order_book'][sym]
                # Widen spreads
                best_bid = book['bid'].peekitem(-1)[0]
                best_ask = book['ask'].peekitem(0)[0]
                spread = best_ask - best_bid
                new_spread = spread * event.parameters['spread_increase']
                
                # Clear order book
                book['bid'] = SortedDict({best_bid - new_spread/2: sum(book['bid'].values())})
                book['ask'] = SortedDict({best_ask + new_spread/2: sum(book['ask'].values())})

    def reset(self) -> Dict:
        """Reset environment to initial state"""
        self.__init__(self.config)
        return self.state

    def render(self, mode='candlestick'):
        """Visualize market state"""
        if mode == 'candlestick':
            self._plot_candlesticks()
        elif mode == 'order_book':
            self._plot_order_book_depth()
        elif mode == 'portfolio':
            self._plot_portfolio_evolution()

    def _init_order_books(self) -> Dict[str, Dict]:
        """Initialize limit order books for all assets"""
        books = {}
        for sym in self.assets:
            # Generate synthetic order book levels
            mid_price = np.random.uniform(50, 200)
            spread = mid_price * 0.0005
            books[sym] = {
                'bid': SortedDict({
                    mid_price - spread*(i+1): np.random.lognormal(3, 0.5)
                    for i in range(5)
                }),
                'ask': SortedDict({
                    mid_price + spread*(i+1): np.random.lognormal(3, 0.5)
                    for i in range(5)
                })
            }
        return books

    def _init_data_system(self, source: str, params: Dict):
        """Initialize market data generator"""
        if source == 'synthetic':
            return self._synthetic_data_generator(params)
        elif source == 'historical':
            return self._historical_data_loader(params)
        
    def _synthetic_data_generator(self, params: Dict):
        """Generate realistic multi-asset price paths"""
        num_assets = len(self.assets)
        cov = params.get('correlation_matrix', np.eye(num_assets)) * params.get('volatility', 0.02)
        
        while True:
            # Generate correlated returns
            returns = multivariate_normal.rvs(
                mean=np.zeros(num_assets),
                cov=cov,
                size=params.get('time_steps', 390)
            )
            
            # Incorporate stochastic volatility
            vol_shocks = np.random.gamma(1, 0.1, size=returns.shape)
            returns *= vol_shocks
            
            # Generate OHLCV data
            yield pd.DataFrame({
                sym: {
                    'open': np.exp(np.cumsum(ret) * 0.01),
                    'high': np.exp(np.cumsum(ret) * 0.01) * 1.005,
                    'low': np.exp(np.cumsum(ret) * 0.01) * 0.995,
                    'close': np.exp(np.cumsum(ret) * 0.01),
                    'volume': np.random.lognormal(10, 2)
                } for sym, ret in zip(self.assets, returns.T)
            })
    
    # Additional 150+ lines of implementation for:
    # - Historical data replay
    # - Order book visualization
    # - Portfolio tracking
    # - Risk management controls
    # - Transaction cost models
    # - Market impact models
    # - Corporate action handling

# Example Configuration
if __name__ == "__main__":
    config = {
        "assets": [
            Asset("AAPL", "EQUITY", 0.01, 100, {"maker": 0.0001, "taker": 0.0003}),
            Asset("ES1", "FUTURE", 0.25, 1, {"maker": 0.00005, "taker": 0.0001}),
            Asset("TSLA", "EQUITY", 0.01, 100, {"maker": 0.0002, "taker": 0.0005})
        ],
        "initial_balance": 1_000_000,
        "position_limits": {
            "AAPL": (-1000, 1000),
            "ES1": (-500, 500),
            "TSLA": (-500, 500)
        },
        "data_source": "synthetic",
        "data_params": {
            "correlation_matrix": [[1.0, 0.3, 0.4],
                                   [0.3, 1.0, 0.2],
                                   [0.4, 0.2, 1.0]],
            "volatility": 0.02
        },
        "reward_components": {
            "sharpe_ratio": 0.7,
            "drawdown_penalty": -0.3,
            "turnover_cost": -0.2
        },
        "market_event_prob": 0.05
    }
    
    env = FinancialMarketEnv(config)
    obs = env.reset()
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render(mode='portfolio')
        
        if done:
            break
