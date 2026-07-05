import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gym import spaces
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "normal")
DEFAULT_DATA_PATH = os.path.join(ROOT, "data", "Data4Fin.csv")


# ----------------------------
# Random seed
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------
# Multi-asset environment
# ----------------------------
class MultiAssetTradingEnv:
    metadata = {'render.modes': ['human']}
    
    def __init__(self, asset_dfs, window_size=5, date_bound=None, trade_fee=0.001, 
                 initial_capital=10000, training=True, buy_ratio=0.2, sell_ratio=0.5):
        super().__init__()
        if date_bound:
            self.start_date, self.end_date = date_bound
            if self.start_date >= self.end_date:
                raise ValueError(f"Invalid date_bound: start {self.start_date} >= end {self.end_date}")
        else:
            self.start_date, self.end_date = None, None
        
        self.asset_dfs = {}
        for name, df in asset_dfs.items():
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
            self.asset_dfs[name] = df.sort_index(ascending=True).copy()
        
        self.asset_names = list(self.asset_dfs.keys())
        self.n_assets = len(self.asset_names)
        
        self.window_size = window_size
        self.trade_fee = trade_fee
        self.initial_capital = initial_capital
        self.training = training
        self.max_total_value = initial_capital
        
        self.buy_ratio = buy_ratio
        self.sell_ratio = sell_ratio
        
        self._filter_data_by_dates()
        
        self.action_space = spaces.Discrete(2 **self.n_assets)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.n_assets, self.n_features),
            dtype=np.float32
        )
        
        self.reset()

    def _filter_data_by_dates(self):
        filtered_dfs = {}
        min_length = float('inf')
        
        for name in self.asset_names:
            df = self.asset_dfs[name]
            
            if self.start_date and self.end_date:
                mask = (df.index >= self.start_date) & (df.index <= self.end_date)
                filtered_df = df.loc[mask].copy()
            else:
                filtered_df = df.copy()
            
            if len(filtered_df) < self.window_size:
                raise ValueError(f"Asset {name}: need >= {self.window_size} rows, got {len(filtered_df)}")
            
            filtered_dfs[name] = filtered_df
            min_length = min(min_length, len(filtered_df))
        
        self.filtered_dfs = {}
        for name in self.asset_names:
            self.filtered_dfs[name] = filtered_dfs[name].iloc[:min_length].copy()
        
        self.features = {}
        self.prices = {}
        self.dates = self.filtered_dfs[self.asset_names[0]].index
        
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        available_features = []
        sample_df = self.filtered_dfs[self.asset_names[0]]
        for feature in self.feature_columns:
            found = False
            for case in [feature, feature.lower(), feature.upper()]:
                if case in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df[case]):
                    available_features.append(case)
                    found = True
                    break
            if found:
                continue
        
        if not available_features:
            print("Warning: indicator columns missing; using OHLCV features only")
            available_features = ['open', 'high', 'low', 'close', 'volume']
        
        self.feature_columns = available_features
        self.n_features = len(self.feature_columns)
        
        all_features = []
        for name in self.asset_names:
            df = self.filtered_dfs[name].copy()
            self.prices[name] = df['close'].values
            
            df['return'] = df['close'].pct_change().fillna(0)
            df['volatility'] = df['high'] - df['low']
            df['price_change'] = (df['close'] - df['open']) / df['open'].replace(0, 1e-8)
            df['volume_change'] = df['volume'].pct_change().fillna(0)
            
            sma5_col = next((col for col in df.columns if 'sma_5' in col.lower()), None)
            sma20_col = next((col for col in df.columns if 'sma_20' in col.lower()), None)
            if sma5_col and sma20_col:
                df['sma_ratio'] = df[sma5_col] / df[sma20_col].replace(0, 1e-8)
                if 'sma_ratio' not in self.feature_columns:
                    self.feature_columns.append('sma_ratio')
            
            upper_bb_col = next((col for col in df.columns if 'upperbb' in col.lower()), None)
            lower_bb_col = next((col for col in df.columns if 'lowerbb' in col.lower()), None)
            if upper_bb_col and lower_bb_col:
                df['bb_width'] = (df[upper_bb_col] - df[lower_bb_col]) / df['close'].replace(0, 1e-8)
                if 'bb_width' not in self.feature_columns:
                    self.feature_columns.append('bb_width')
            
            valid_features = []
            for f in self.feature_columns:
                if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                    valid_features.append(f)
                else:
                    df[f] = 0.0
                    valid_features.append(f)
            
            feature_data = df[valid_features].values.astype(np.float32)
            feature_data = np.nan_to_num(feature_data)
            
            for col in range(feature_data.shape[1]):
                col_data = feature_data[:, col]
                mean = np.mean(col_data)
                std = np.std(col_data) + 1e-8
                feature_data[:, col] = (col_data - mean) / std
            
            all_features.append(feature_data)
            self.features[name] = feature_data
        
        feature_counts = [f.shape[1] for f in all_features]
        if len(set(feature_counts)) > 1:
            max_features = max(feature_counts)
            for name in self.asset_names:
                if self.features[name].shape[1] < max_features:
                    padding = np.zeros((self.features[name].shape[0], max_features - self.features[name].shape[1]))
                    self.features[name] = np.hstack((self.features[name], padding))
            self.n_features = max_features
        else:
            self.n_features = feature_counts[0]


    def _get_state(self):
        state = []
        target_shape = None
        
        for name in self.asset_names:
            start = self.current_tick
            end = self.current_tick + self.window_size
            
            if end > len(self.features[name]):
                end = len(self.features[name])
                asset_state = self.features[name][start:end].copy()
                pad_length = self.window_size - len(asset_state)
                if pad_length > 0:
                    last_value = asset_state[-1:] if len(asset_state) > 0 else np.zeros((1, self.n_features))
                    asset_state = np.vstack([asset_state] + [last_value] * pad_length)
            else:
                asset_state = self.features[name][start:end].copy()
            
            asset_state = np.nan_to_num(asset_state)
            
            if target_shape is None:
                target_shape = asset_state.shape
            else:
                if asset_state.shape != target_shape:
                    asset_state = np.resize(asset_state, target_shape)
            
            state.append(asset_state)
        
        shapes = [s.shape for s in state]
        if len(set(shapes)) > 1:
            min_shape = min(shapes, key=lambda x: x[0])
            state = [s[:min_shape[0], :min_shape[1]] for s in state]
        
        state_np = np.stack(state, axis=1).astype(np.float32)
        return state_np
 
    def _get_min_length(self):
        return min(len(df) for df in self.filtered_dfs.values())
    
    def _decode_action(self, action):
        return np.array([(action >> i) & 1 for i in range(self.n_assets)], dtype=int)
    
    def reset(self):
        self.current_tick = 0
        self.positions = np.zeros(self.n_assets, dtype=int)
        self.buy_prices = np.zeros(self.n_assets)
        self.capital = self.initial_capital
        self.total_assets_value = 0
        self.total_profit = 0
        self.transactions = [] if not self.training else None
        self.hold_steps = 0
        self.max_total_value = self.initial_capital
        return self._get_state()
    
    def step(self, action):
        current_date = self.dates[self.current_tick] if self.current_tick < len(self.dates) else None
        reward = 0
        self.hold_steps += 1
        
        if self.training and self.hold_steps >= 50:
            rand_asset = np.random.randint(self.n_assets)
            action_array = self._decode_action(action)
            action_array[rand_asset] = 1 - self.positions[rand_asset]
            action = sum([(v << i) for i, v in enumerate(action_array)])
            self.hold_steps = 0
        
        if hasattr(self, 'hold_penalty_threshold') and self.hold_steps > self.hold_penalty_threshold:
            penalty = self.hold_penalty_factor * np.log1p(self.hold_steps - self.hold_penalty_threshold)
            reward -= penalty
            if hasattr(self, 'hold_severe_threshold') and self.hold_steps > self.hold_severe_threshold:
                reward -= self.hold_severe_penalty
        
        action_array = self._decode_action(action)
        
        prev_total_value = self.capital + self.total_assets_value
        
        for i, name in enumerate(self.asset_names):
            current_price = self.prices[name][self.current_tick]
            action_i = action_array[i]
            
            if action_i == 1 and self.positions[i] == 0:
                max_investment = self.capital * self.buy_ratio
                buy_cost_per_share = current_price * (1 + self.trade_fee)
                
                if buy_cost_per_share > 0 and max_investment >= buy_cost_per_share:
                    shares_to_buy = int(max_investment / buy_cost_per_share)
                    
                    if shares_to_buy > 0:
                        total_cost = shares_to_buy * buy_cost_per_share
                        self.positions[i] = shares_to_buy
                        self.buy_prices[i] = buy_cost_per_share
                        self.capital -= total_cost
                        
                        if not self.training:
                            self.transactions.append({
                                "date": current_date, "asset": name, "price": current_price,
                                "action": "buy", "shares": shares_to_buy, "profit": 0
                            })
                        if hasattr(self, 'buy_reward'):
                            reward += self.buy_reward
                        self.hold_steps = 0
                else:
                    if hasattr(self, 'insufficient_fund_penalty'):
                        reward -= self.insufficient_fund_penalty
            
            elif action_i == 0 and self.positions[i] > 0:
                shares_to_sell = int(self.positions[i] * self.sell_ratio)
                
                if shares_to_sell > 0:
                    sell_price_per_share = current_price * (1 - self.trade_fee)
                    total_revenue = shares_to_sell * sell_price_per_share
                    cost_basis = shares_to_sell * self.buy_prices[i]
                    profit = total_revenue - cost_basis
                    self.total_profit += profit
                    
                    self.positions[i] -= shares_to_sell
                    self.capital += total_revenue
                    
                    if not self.training:
                        self.transactions.append({
                            "date": current_date, "asset": name, "price": current_price,
                            "action": "sell", "shares": shares_to_sell, "profit": profit
                        })
                    
                    if hasattr(self, 'profit_reward_factor'):
                        profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0
                        reward += profit_pct * self.profit_reward_factor
                        if profit > 0:
                            reward += self.profit_reward_factor * 2
                    
                    self.hold_steps = 0
        
        self.total_assets_value = sum(
            self.positions[i] * self.prices[name][self.current_tick] 
            for i, name in enumerate(self.asset_names)
        )
        
        current_total_value = self.capital + self.total_assets_value
        if hasattr(self, 'value_increase_factor'):
            value_change = (current_total_value - prev_total_value) 
            reward += value_change * self.value_increase_factor
        
        if current_total_value > self.max_total_value:
            reward += 5.0
            self.max_total_value = current_total_value
        
        invalid_actions = sum(
            (action_array[i] == 1 and self.positions[i] > 0) or (action_array[i] == 0 and self.positions[i] == 0)
            for i in range(self.n_assets)
        )
        if hasattr(self, 'invalid_action_penalty'):
            reward -= invalid_actions * self.invalid_action_penalty
        
        reward = np.clip(reward, -10.0, 10.0)
        
        self.current_tick += 1
        done = self.current_tick >= len(self.dates) - self.window_size
        
        info = {
            "total_profit": self.total_profit,
            "total_value": current_total_value,
            "capital": self.capital,
            "positions": {name: self.positions[i] for i, name in enumerate(self.asset_names)},
            "date": current_date,
            "action": action,
            "reward": reward
        }
        
        return self._get_state(), reward, done, info


# ----------------------------
# Models
# ----------------------------
class A2CNetwork(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_dim=128):
        super().__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(state_shape[2], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape[::-1])
            self.feature_size = self.feature_extractor(dummy_input).shape[1]
        
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)
        
        if torch.isnan(features).any():
            features = torch.nan_to_num(features)
        
        actor_out = self.actor(features)
        critic_out = self.critic(features)
        
        actor_out = torch.clamp(actor_out, 1e-8, 1.0 - 1e-8)
        actor_out = actor_out / actor_out.sum(dim=-1, keepdim=True)
        
        return actor_out, critic_out
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            action_probs, state_value = self.forward(state)
            
            if torch.isnan(action_probs).any():
                action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
            
            dist = Categorical(action_probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action).detach(), state_value.item()


class LSTMNetwork(nn.Module):
    """LSTM policy for trading"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return self.softmax(out)
    
    def get_action(self, state):
        state_reshaped = state.reshape(1, state.shape[0], -1)
        state_tensor = torch.FloatTensor(state_reshaped).to(self.device)
        
        with torch.no_grad():
            action_probs = self.forward(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item(), dist.log_prob(action).detach(), 0


class CNNNetwork(nn.Module):
    """CNN policy for trading"""
    def __init__(self, state_shape, action_dim, hidden_dim=128):
        super().__init__()
        self.state_shape = state_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(state_shape[2], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_shape[::-1])
            self.feature_size = self.conv_layers(dummy_input).shape[1]
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        features = self.conv_layers(x)
        return self.fc_layers(features)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item(), dist.log_prob(action).detach(), 0


class ANNNetwork(nn.Module):
    """MLP policy for trading"""
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def get_action(self, state):
        state = state.flatten()
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        with torch.no_grad():
            action_probs = self.forward(state)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        return action.item(), dist.log_prob(action).detach(), 0


# ----------------------------
# Training
# ----------------------------
def train_a2c(asset_dfs, window_size, train_dates, test_dates, hyperparams, 
             output_folder=None, model_filename_base='a2c_multi_model'):
    total_timesteps = hyperparams['a2c_total_timesteps']
    train_years = (train_dates[1] - train_dates[0]).days / 365.25
    test_year_str = test_dates[1].year
    model_filename = f"{model_filename_base}_{total_timesteps//1000}k_{train_years:.0f}y_{test_year_str}.pth"
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    
    test_metrics = None
    
    if is_main_process:
        print(f"Training with A2C distributed, Device: {device}")
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, model_filename)
        loss_path = os.path.join(output_folder, f'a2c_loss_{total_timesteps//1000}k.png') if output_folder else None
        strategy_plot_path = os.path.join(output_folder, f'a2c_strategy.png') if output_folder else None
    
    train_env = MultiAssetTradingEnv(
        asset_dfs=asset_dfs, window_size=window_size, date_bound=train_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=True, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    train_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    train_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    train_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    train_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    train_env.buy_reward = hyperparams['buy_reward']
    train_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    train_env.profit_reward_factor = hyperparams['profit_reward_factor']
    train_env.value_increase_factor = hyperparams['value_increase_factor']
    train_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    env_length = len(train_env.dates) - window_size
    if env_length <= 0:
        if is_main_process:
            print("Error: Training environment length is insufficient")
        return {"overall": {"total_profit": 0, "overall_return_pct": 0, "total_trades": 0}, "assets": {}}, None
    
    max_rollout_steps = min(hyperparams['a2c_rollout_steps'], env_length - 10)
    if is_main_process and max_rollout_steps < hyperparams['a2c_rollout_steps']:
        print(f"Adjusting rollout_steps to {max_rollout_steps}")
    
    state_shape = train_env.observation_space.shape
    action_dim = train_env.action_space.n
    model = A2CNetwork(state_shape, action_dim, hidden_dim=hyperparams['a2c_hidden_dim'])
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['a2c_lr'])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    
    a2c_params = {
        "gamma": hyperparams['a2c_gamma'],
        "entropy_coef": hyperparams['a2c_entropy_coef'],
        "value_coef": hyperparams['a2c_value_coef'],
        "rollout_steps": max_rollout_steps,
        "num_updates": hyperparams['a2c_total_timesteps'] // max_rollout_steps
    }
    
    if is_main_process:
        print(f"Starting A2C training, Total timesteps: {hyperparams['a2c_total_timesteps']}")
        print(f"Training date range: {train_env.dates[0]} to {train_env.dates[-1]}")
        loss_history = []
        action_history = []
        reward_history = []
        pbar = tqdm(total=a2c_params["num_updates"], desc="A2C Training")
    
    total_steps = 0
    for update in range(a2c_params["num_updates"]):
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        state = train_env.reset()
        
        for _ in range(a2c_params["rollout_steps"]):
            action, log_prob, value = model.get_action(state)
            next_state, reward, done, info = train_env.step(action)
            
            if is_main_process:
                action_history.append(action)
                reward_history.append(reward)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(1 - done)
            
            state = next_state
            total_steps += 1
            if done:
                state = train_env.reset()
        
        returns = []
        advantages = []
        R = 0 if done else model.get_action(state)[2]
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + a2c_params["gamma"] * dones[i] * R
            returns.insert(0, R)
            advantages.insert(0, R - values[i])
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        log_probs = torch.cat(log_probs).to(device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        action_probs, state_values = model(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = F.mse_loss(state_values.squeeze(), returns)
        entropy_loss = -dist.entropy().mean()
        
        total_loss = policy_loss + a2c_params["value_coef"] * value_loss + a2c_params["entropy_coef"] * entropy_loss
        
        optimizer.zero_grad()
        accelerator.backward(total_loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        if is_main_process:
            loss_history.append(total_loss.item())
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", 
                             "MeanReward": f"{np.mean(rewards):.2f}",
                             "LR": f"{scheduler.get_last_lr()[0]:.6f}"})
        
        if total_steps >= hyperparams['a2c_total_timesteps']:
            break
    
    if is_main_process:
        pbar.close()
        plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        
        if action_history:
            plt.figure(figsize=(12, 5))
            plt.hist(action_history, bins=min(20, action_dim), alpha=0.7)
            plt.title('Action Distribution During Training')
            plt.xlabel('Action')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_folder, f'a2c_action_dist.png'))
            plt.close()
        
        if reward_history:
            plt.figure(figsize=(12, 5))
            plt.plot(pd.Series(reward_history).rolling(window=100).mean())
            plt.title('Smoothed Reward During Training')
            plt.xlabel('Steps')
            plt.ylabel('Reward (Smoothed)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_folder, f'a2c_reward.png'))
            plt.close()
        
        if loss_history:
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history)
            plt.title('A2C Training Loss')
            plt.xlabel('Update Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(loss_path)
            plt.close()
            pd.Series(loss_history).to_csv(os.path.join(output_folder, f'a2c_loss_history.csv'), index=False)        
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), model_path)
        print(f"A2C model saved to {model_path}")
        
        test_metrics = evaluate_a2c(
            asset_dfs, 
            model_path, 
            state_shape, 
            action_dim,
            window_size,
            test_dates,
            hyperparams,
            n_runs=3,
            output_folder=output_folder,
            strategy_plot_path=strategy_plot_path
        )
    else:
        test_metrics = None
    
    accelerator.wait_for_everyone()
    return test_metrics, model if is_main_process else None


def train_lstm(asset_dfs, window_size, train_dates, test_dates, hyperparams, 
              output_folder=None, model_filename_base='lstm_multi_model'):
    """Train LSTM policy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM on {device}")
    
    train_env = MultiAssetTradingEnv(
        asset_dfs=asset_dfs, window_size=window_size, date_bound=train_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=True, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    train_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    train_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    train_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    train_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    train_env.buy_reward = hyperparams['buy_reward']
    train_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    train_env.profit_reward_factor = hyperparams['profit_reward_factor']
    train_env.value_increase_factor = hyperparams['value_increase_factor']
    train_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    state_shape = train_env.observation_space.shape
    action_dim = train_env.action_space.n
    input_dim = state_shape[1] * state_shape[2]
    
    model = LSTMNetwork(
        input_dim=input_dim,
        hidden_dim=hyperparams['lstm_hidden_dim'],
        output_dim=action_dim,
        num_layers=hyperparams['lstm_num_layers']
    ).to(device)
    model.device = device
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lstm_lr'])
    criterion = nn.CrossEntropyLoss()
    
    total_timesteps = hyperparams['lstm_total_timesteps']
    batch_size = hyperparams['lstm_batch_size']
    episode_length = hyperparams['lstm_episode_length']
    
    loss_history = []
    pbar = tqdm(total=total_timesteps, desc="LSTM Training")
    total_steps = 0
    
    while total_steps < total_timesteps:
        state = train_env.reset()
        episode_loss = 0
        steps_in_episode = 0
        
        while steps_in_episode < episode_length:
            action, _, _ = model.get_action(state)
            
            next_state, reward, done, info = train_env.step(action)
            
            state_reshaped = state.reshape(1, state.shape[0], -1)
            state_tensor = torch.FloatTensor(state_reshaped).to(device)
            target = torch.tensor([action]).to(device)
            
            outputs = model(state_tensor)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            episode_loss += loss.item()
            state = next_state
            total_steps += 1
            steps_in_episode += 1
            pbar.update(1)
            
            if done or total_steps >= total_timesteps:
                break
        
        loss_history.append(episode_loss / max(steps_in_episode, 1))
    
    pbar.close()
    
    if output_folder:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'{model_filename_base}.pth')
        torch.save(model.state_dict(), model_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('LSTM Training Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'lstm_loss.png'))
        plt.close()
    
    test_metrics = evaluate_lstm(
        asset_dfs, model_path, state_shape, action_dim, input_dim,
        window_size, test_dates, hyperparams, output_folder=output_folder
    )
    
    return test_metrics, model


def train_cnn(asset_dfs, window_size, train_dates, test_dates, hyperparams, 
             output_folder=None, model_filename_base='cnn_multi_model'):
    """Train CNN policy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN on {device}")
    
    train_env = MultiAssetTradingEnv(
        asset_dfs=asset_dfs, window_size=window_size, date_bound=train_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=True, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    train_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    train_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    train_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    train_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    train_env.buy_reward = hyperparams['buy_reward']
    train_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    train_env.profit_reward_factor = hyperparams['profit_reward_factor']
    train_env.value_increase_factor = hyperparams['value_increase_factor']
    train_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    state_shape = train_env.observation_space.shape
    action_dim = train_env.action_space.n
    
    model = CNNNetwork(
        state_shape=state_shape,
        action_dim=action_dim,
        hidden_dim=hyperparams['cnn_hidden_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['cnn_lr'])
    criterion = nn.CrossEntropyLoss()
    
    total_timesteps = hyperparams['cnn_total_timesteps']
    episode_length = hyperparams['cnn_episode_length']
    
    loss_history = []
    pbar = tqdm(total=total_timesteps, desc="CNN Training")
    total_steps = 0
    
    while total_steps < total_timesteps:
        state = train_env.reset()
        episode_loss = 0
        steps_in_episode = 0
        
        while steps_in_episode < episode_length:
            action, _, _ = model.get_action(state)
            
            next_state, reward, done, info = train_env.step(action)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            target = torch.tensor([action]).to(device)
            
            outputs = model(state_tensor)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            episode_loss += loss.item()
            state = next_state
            total_steps += 1
            steps_in_episode += 1
            pbar.update(1)
            
            if done or total_steps >= total_timesteps:
                break
        
        loss_history.append(episode_loss / max(steps_in_episode, 1))
    
    pbar.close()
    
    if output_folder:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'{model_filename_base}.pth')
        torch.save(model.state_dict(), model_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('CNN Training Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'cnn_loss.png'))
        plt.close()
    
    test_metrics = evaluate_cnn(
        asset_dfs, model_path, state_shape, action_dim,
        window_size, test_dates, hyperparams, output_folder=output_folder
    )
    
    return test_metrics, model


def train_ann(asset_dfs, window_size, train_dates, test_dates, hyperparams, 
             output_folder=None, model_filename_base='ann_multi_model'):
    """Train MLP policy"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training ANN on {device}")
    
    train_env = MultiAssetTradingEnv(
        asset_dfs=asset_dfs, window_size=window_size, date_bound=train_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=True, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    train_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    train_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    train_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    train_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    train_env.buy_reward = hyperparams['buy_reward']
    train_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    train_env.profit_reward_factor = hyperparams['profit_reward_factor']
    train_env.value_increase_factor = hyperparams['value_increase_factor']
    train_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    state_shape = train_env.observation_space.shape
    action_dim = train_env.action_space.n
    input_dim = state_shape[0] * state_shape[1] * state_shape[2]
    
    model = ANNNetwork(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hyperparams['ann_hidden_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['ann_lr'])
    criterion = nn.CrossEntropyLoss()
    
    total_timesteps = hyperparams['ann_total_timesteps']
    episode_length = hyperparams['ann_episode_length']
    
    loss_history = []
    pbar = tqdm(total=total_timesteps, desc="ANN Training")
    total_steps = 0
    
    while total_steps < total_timesteps:
        state = train_env.reset()
        episode_loss = 0
        steps_in_episode = 0
        
        while steps_in_episode < episode_length:
            action, _, _ = model.get_action(state)
            
            next_state, reward, done, info = train_env.step(action)
            
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
            target = torch.tensor([action]).to(device)
            
            outputs = model(state_tensor)
            loss = criterion(outputs, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            episode_loss += loss.item()
            state = next_state
            total_steps += 1
            steps_in_episode += 1
            pbar.update(1)
            
            if done or total_steps >= total_timesteps:
                break
        
        loss_history.append(episode_loss / max(steps_in_episode, 1))
    
    pbar.close()
    
    if output_folder:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'{model_filename_base}.pth')
        torch.save(model.state_dict(), model_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('ANN Training Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, 'ann_loss.png'))
        plt.close()
    
    test_metrics = evaluate_ann(
        asset_dfs, model_path, state_shape, action_dim, input_dim,
        window_size, test_dates, hyperparams, output_folder=output_folder
    )
    
    return test_metrics, model


def run_arima_strategy(asset_dfs, window_size, test_dates, hyperparams, output_folder=None):
    """Run ARIMA baseline with tqdm"""
    print("Running ARIMA strategy...")
    
    np.random.seed(hyperparams['seed'])
    
    test_env = MultiAssetTradingEnv(
        asset_dfs, window_size=window_size, date_bound=test_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=False, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    test_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    test_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    test_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    test_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    test_env.buy_reward = hyperparams['buy_reward']
    test_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    test_env.profit_reward_factor = hyperparams['profit_reward_factor']
    test_env.value_increase_factor = hyperparams['value_increase_factor']
    test_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    total_steps = len(test_env.dates) - window_size
    pbar = tqdm(total=total_steps, desc="ARIMA Inference")
    
    state = test_env.reset()
    portfolio_values = [hyperparams['initial_capital']]
    
    step_count = 0
    while True:
        current_prices = [test_env.prices[name][test_env.current_tick] for name in test_env.asset_names]
        
        actions = []
        for i, name in enumerate(test_env.asset_names):
            price_history = test_env.prices[name][max(0, test_env.current_tick-50):test_env.current_tick]
            
            if len(price_history) < 10:
                actions.append(0)
                continue
            
            try:
                model = ARIMA(price_history, order=hyperparams['arima_order'])
                model_fit = model.fit()
                
                forecast = model_fit.forecast(steps=1)[0]
                
                current_price = price_history[-1]
                if forecast > current_price * (1 + hyperparams['arima_threshold']):
                    actions.append(1)
                else:
                    actions.append(0)
            except:
                actions.append(0)
        
        action = sum([(v << i) for i, v in enumerate(actions)])
        
        next_state, reward, done, info = test_env.step(action)
        
        portfolio_value = info['total_value']
        portfolio_values.append(portfolio_value)
        
        state = next_state
        step_count += 1
        pbar.update(1)
        
        if done or step_count >= total_steps:
            break
    
    pbar.close()
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    
    total_profit = test_env.total_profit
    print(f"ARIMA run - Profit: {total_profit:.2f}")
    
    analyzer = MultiAssetTradingAnalyzer()
    metrics = analyzer.calculate_metrics(test_env.transactions, hyperparams['initial_capital'])
    
    strategy_metrics = calculate_strategy_metrics(daily_returns)
    
    if output_folder:
        MultiAssetTradingAnalyzer.visualize(
            test_env, 
            f"ARIMA Strategy Trading Results (Multi-Asset, Profit: {total_profit:.2f})", 
            os.path.join(output_folder, 'arima_strategy.png')
        )
        
        transactions_df = pd.DataFrame(test_env.transactions)
        transactions_df.to_csv(os.path.join(output_folder, 'arima_transactions.csv'), index=False)
    
    result = {
        "total_profit": total_profit,
        "final_value": hyperparams['initial_capital'] + total_profit,
        "transactions": test_env.transactions,
        "overall": metrics["overall"],
        "assets": metrics["assets"],
        "strategy_metrics": strategy_metrics,
        "daily_returns": daily_returns
    }
    
    return result


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_a2c(asset_dfs, model_path, state_shape, action_dim, window_size, test_dates, 
                hyperparams, n_runs=3, output_folder=None, strategy_plot_path=None):
    """Evaluate A2C and return strategy metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_profits = []
    all_transactions = []
    final_values = []
    all_daily_returns = []
    
    test_env = MultiAssetTradingEnv(
        asset_dfs, window_size=window_size, date_bound=test_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=False, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    test_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    test_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    test_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    test_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    test_env.buy_reward = hyperparams['buy_reward']
    test_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    test_env.profit_reward_factor = hyperparams['profit_reward_factor']
    test_env.value_increase_factor = hyperparams['value_increase_factor']
    test_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    print(f"Testing date range: {test_env.dates[0]} to {test_env.dates[-1]}")
    print(f"Running {n_runs} evaluation runs...")
    
    model = A2CNetwork(state_shape, action_dim, hidden_dim=hyperparams['a2c_hidden_dim'])
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    for run in range(n_runs):
        obs = test_env.reset()
        portfolio_values = [hyperparams['initial_capital']]
        
        while True:
            with torch.no_grad():
                action, _, _ = model.get_action(obs)
            obs, reward, done, info = test_env.step(action)
            portfolio_values.append(info['total_value'])
            
            if done:
                break
        
        run_returns = (np.array(portfolio_values[1:]) - np.array(portfolio_values[:-1])) / np.array(portfolio_values[:-1])
        all_daily_returns.append(run_returns)
        
        total_profits.append(test_env.total_profit)
        final_values.append(test_env.capital + test_env.total_assets_value)
        all_transactions.extend(test_env.transactions)
        print(f"A2C run {run+1}/{n_runs} - Profit: {test_env.total_profit:.2f}, Final Value: {final_values[-1]:.2f}")
    
    avg_profit = np.mean(total_profits)
    std_profit = np.std(total_profits)
    avg_final_value = np.mean(final_values)
    
    min_length = min(len(returns) for returns in all_daily_returns)
    trimmed_returns = [returns[:min_length] for returns in all_daily_returns]
    avg_daily_returns = np.mean(trimmed_returns, axis=0)
    
    print(f"Average Profit across {n_runs} runs: {avg_profit:.2f} (±{std_profit:.2f})")
    
    MultiAssetTradingAnalyzer.visualize(
        test_env, 
        f"A2C Strategy Trading Results (Multi-Asset, Avg Profit: {avg_profit:.2f})", 
        strategy_plot_path
    )
    
    analyzer = MultiAssetTradingAnalyzer()
    metrics = analyzer.calculate_metrics(all_transactions, hyperparams['initial_capital'])
    
    strategy_metrics = calculate_strategy_metrics(avg_daily_returns)
    
    return {
        "total_profit": avg_profit,
        "final_value": avg_final_value,
        "transactions": all_transactions,
        "overall": metrics["overall"],
        "assets": metrics["assets"],
        "std_profit": std_profit,
        "strategy_metrics": strategy_metrics,
        "daily_returns": avg_daily_returns
    }


def evaluate_lstm(asset_dfs, model_path, state_shape, action_dim, input_dim, window_size, 
                 test_dates, hyperparams, output_folder=None):
    """Evaluate LSTM"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(hyperparams['seed'])
    
    test_env = MultiAssetTradingEnv(
        asset_dfs, window_size=window_size, date_bound=test_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=False, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    test_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    test_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    test_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    test_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    test_env.buy_reward = hyperparams['buy_reward']
    test_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    test_env.profit_reward_factor = hyperparams['profit_reward_factor']
    test_env.value_increase_factor = hyperparams['value_increase_factor']
    test_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    model = LSTMNetwork(
        input_dim=input_dim,
        hidden_dim=hyperparams['lstm_hidden_dim'],
        output_dim=action_dim,
        num_layers=hyperparams['lstm_num_layers']
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.device = device
    model.eval()
    
    obs = test_env.reset()
    portfolio_values = [hyperparams['initial_capital']]
    
    while True:
        with torch.no_grad():
            action, _, _ = model.get_action(obs)
        
        obs, reward, done, info = test_env.step(action)
        portfolio_values.append(info['total_value'])
        
        if done:
            break
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    
    total_profit = test_env.total_profit
    print(f"LSTM run - Profit: {total_profit:.2f}")
    
    analyzer = MultiAssetTradingAnalyzer()
    metrics = analyzer.calculate_metrics(test_env.transactions, hyperparams['initial_capital'])
    
    strategy_metrics = calculate_strategy_metrics(daily_returns)
    
    if output_folder:
        MultiAssetTradingAnalyzer.visualize(
            test_env, 
            f"LSTM Strategy Trading Results (Multi-Asset, Profit: {total_profit:.2f})", 
            os.path.join(output_folder, 'lstm_strategy.png')
        )
    
    result = {
        "total_profit": total_profit,
        "final_value": hyperparams['initial_capital'] + total_profit,
        "transactions": test_env.transactions,
        "overall": metrics["overall"],
        "assets": metrics["assets"],
        "strategy_metrics": strategy_metrics,
        "daily_returns": daily_returns
    }
    
    return result


def evaluate_cnn(asset_dfs, model_path, state_shape, action_dim, window_size, 
                test_dates, hyperparams, output_folder=None):
    """Evaluate CNN"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(hyperparams['seed'])
    
    test_env = MultiAssetTradingEnv(
        asset_dfs, window_size=window_size, date_bound=test_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=False, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    test_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    test_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    test_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    test_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    test_env.buy_reward = hyperparams['buy_reward']
    test_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    test_env.profit_reward_factor = hyperparams['profit_reward_factor']
    test_env.value_increase_factor = hyperparams['value_increase_factor']
    test_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    model = CNNNetwork(
        state_shape=state_shape,
        action_dim=action_dim,
        hidden_dim=hyperparams['cnn_hidden_dim']
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    obs = test_env.reset()
    portfolio_values = [hyperparams['initial_capital']]
    
    while True:
        with torch.no_grad():
            action, _, _ = model.get_action(obs)
        
        obs, reward, done, info = test_env.step(action)
        portfolio_values.append(info['total_value'])
        
        if done:
            break
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    
    total_profit = test_env.total_profit
    print(f"CNN run - Profit: {total_profit:.2f}")
    
    analyzer = MultiAssetTradingAnalyzer()
    metrics = analyzer.calculate_metrics(test_env.transactions, hyperparams['initial_capital'])
    
    strategy_metrics = calculate_strategy_metrics(daily_returns)
    
    if output_folder:
        MultiAssetTradingAnalyzer.visualize(
            test_env, 
            f"CNN Strategy Trading Results (Multi-Asset, Profit: {total_profit:.2f})", 
            os.path.join(output_folder, 'cnn_strategy.png')
        )
    
    result = {
        "total_profit": total_profit,
        "final_value": hyperparams['initial_capital'] + total_profit,
        "transactions": test_env.transactions,
        "overall": metrics["overall"],
        "assets": metrics["assets"],
        "strategy_metrics": strategy_metrics,
        "daily_returns": daily_returns
    }
    
    return result


def evaluate_ann(asset_dfs, model_path, state_shape, action_dim, input_dim, window_size, 
                test_dates, hyperparams, output_folder=None):
    """Evaluate MLP"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    set_seed(hyperparams['seed'])
    
    test_env = MultiAssetTradingEnv(
        asset_dfs, window_size=window_size, date_bound=test_dates,
        trade_fee=hyperparams['trade_fee'], initial_capital=hyperparams['initial_capital'], 
        training=False, buy_ratio=hyperparams['buy_ratio'], sell_ratio=hyperparams['sell_ratio']
    )
    
    test_env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
    test_env.hold_penalty_factor = hyperparams['hold_penalty_factor']
    test_env.hold_severe_threshold = hyperparams['hold_severe_threshold']
    test_env.hold_severe_penalty = hyperparams['hold_severe_penalty']
    test_env.buy_reward = hyperparams['buy_reward']
    test_env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
    test_env.profit_reward_factor = hyperparams['profit_reward_factor']
    test_env.value_increase_factor = hyperparams['value_increase_factor']
    test_env.invalid_action_penalty = hyperparams['invalid_action_penalty']
    
    model = ANNNetwork(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hyperparams['ann_hidden_dim']
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    obs = test_env.reset()
    portfolio_values = [hyperparams['initial_capital']]
    
    while True:
        with torch.no_grad():
            action, _, _ = model.get_action(obs)
        
        obs, reward, done, info = test_env.step(action)
        portfolio_values.append(info['total_value'])
        
        if done:
            break
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]
    
    total_profit = test_env.total_profit
    print(f"ANN run - Profit: {total_profit:.2f}")
    
    analyzer = MultiAssetTradingAnalyzer()
    metrics = analyzer.calculate_metrics(test_env.transactions, hyperparams['initial_capital'])
    
    strategy_metrics = calculate_strategy_metrics(daily_returns)
    
    if output_folder:
        MultiAssetTradingAnalyzer.visualize(
            test_env, 
            f"ANN Strategy Trading Results (Multi-Asset, Profit: {total_profit:.2f})", 
            os.path.join(output_folder, 'ann_strategy.png')
        )
    
    result = {
        "total_profit": total_profit,
        "final_value": hyperparams['initial_capital'] + total_profit,
        "transactions": test_env.transactions,
        "overall": metrics["overall"],
        "assets": metrics["assets"],
        "strategy_metrics": strategy_metrics,
        "daily_returns": daily_returns
    }
    
    return result


# ----------------------------
# Random baseline
# ----------------------------
def run_random_strategy(asset_dfs, window_size=5, test_dates=None, hyperparams=None, output_folder=None):
    """Random baseline over action bitmask"""
    n_runs = 3
    total_profits = []
    all_transactions = []
    all_daily_returns = []
    
    print(f"Running random strategy with {n_runs} runs for averaging...")
    
    test_env = MultiAssetTradingEnv(
        asset_dfs, 
        window_size=window_size, 
        date_bound=test_dates,
        trade_fee=hyperparams['trade_fee'],
        initial_capital=hyperparams['initial_capital'],
        training=False,
        buy_ratio=hyperparams['buy_ratio'],
        sell_ratio=hyperparams['sell_ratio']
    )
    total_steps = len(test_env.dates) - window_size
    
    for run in range(n_runs):
        env = MultiAssetTradingEnv(
            asset_dfs, 
            window_size=window_size, 
            date_bound=test_dates,
            trade_fee=hyperparams['trade_fee'],
            initial_capital=hyperparams['initial_capital'],
            training=False,
            buy_ratio=hyperparams['buy_ratio'],
            sell_ratio=hyperparams['sell_ratio']
        )
        env.hold_penalty_threshold = hyperparams['hold_penalty_threshold']
        env.hold_penalty_factor = hyperparams['hold_penalty_factor']
        env.hold_severe_threshold = hyperparams['hold_severe_threshold']
        env.hold_severe_penalty = hyperparams['hold_severe_penalty']
        env.buy_reward = hyperparams['buy_reward']
        env.insufficient_fund_penalty = hyperparams['insufficient_fund_penalty']
        env.profit_reward_factor = hyperparams['profit_reward_factor']
        env.value_increase_factor = hyperparams['value_increase_factor']
        env.invalid_action_penalty = hyperparams['invalid_action_penalty']
        
        state = env.reset()
        portfolio_values = [hyperparams['initial_capital']]
        
        pbar = tqdm(total=total_steps, desc=f"Random Run {run+1}/{n_runs}")
        step_count = 0
        
        while True:
            action = np.random.randint(0, env.action_space.n)
            state, reward, done, info = env.step(action)
            portfolio_values.append(info['total_value'])
            
            step_count += 1
            pbar.update(1)
            
            if done or step_count >= total_steps:
                break
        
        pbar.close()
        
        run_returns = (np.array(portfolio_values[1:]) - np.array(portfolio_values[:-1])) / np.array(portfolio_values[:-1])
        all_daily_returns.append(run_returns)
        
        total_profits.append(env.total_profit)
        all_transactions.extend(env.transactions)
        print(f"Random run {run+1}/{n_runs} - Profit: {env.total_profit:.2f}")
    
    avg_profit = np.mean(total_profits)
    std_profit = np.std(total_profits)
    
    min_length = min(len(returns) for returns in all_daily_returns)
    trimmed_returns = [returns[:min_length] for returns in all_daily_returns]
    avg_daily_returns = np.mean(trimmed_returns, axis=0)
    
    start_date = test_env.dates[0]
    end_date = test_env.dates[-1]
    print(f"Random strategy average profit over {n_runs} runs: {avg_profit:.2f} (±{std_profit:.2f})")
    
    analyzer = MultiAssetTradingAnalyzer()
    metrics = analyzer.calculate_metrics(all_transactions, hyperparams['initial_capital'])
    
    strategy_metrics = calculate_strategy_metrics(avg_daily_returns)
    
    if output_folder:
        transactions_df = pd.DataFrame(all_transactions)
        transactions_df.to_csv(os.path.join(output_folder, 'random_transactions.csv'), index=False)
        analyzer.visualize(
            test_env, 
            f"Random Strategy Trading Results (Multi-Asset, Avg Profit: {avg_profit:.2f})", 
            os.path.join(output_folder, 'random_strategy.png')
        )
    
    return {
        "total_profit": avg_profit,
        "final_value": hyperparams['initial_capital'] + avg_profit,
        "transactions": all_transactions,
        "overall": metrics["overall"],
        "assets": metrics["assets"],
        "std_profit": std_profit,
        "strategy_metrics": strategy_metrics,
        "daily_returns": avg_daily_returns
    }


# ----------------------------
# Strategy metrics
# ----------------------------
def calculate_strategy_metrics(strategy_returns):
    """
    Compute Tr, Sr, Vol, Mdd from daily strategy returns.

    Args:
        strategy_returns: daily return series

    Returns:
        metric dict
    """
    if len(strategy_returns) == 0:
        return {'Tr': 0, 'Sr': 0, 'Vol': 0, 'Mdd': 0}
    
    strategy_returns_series = pd.Series(strategy_returns)
    
    cumulative_return = (1 + strategy_returns).prod() - 1
    tr = cumulative_return * 100
    
    vol = strategy_returns.std() * np.sqrt(252) * 100
    
    if vol > 0:
        sr = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252))
    else:
        sr = 0
    
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = pd.Series(cumulative_returns).expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    mdd = drawdown.min() * 100
    
    return {
        'Tr': round(tr, 2),
        'Sr': round(sr, 2),
        'Vol': round(vol, 2),
        'Mdd': round(mdd, 2)
    }


# ----------------------------
# Backtest analyzer
# ----------------------------
class MultiAssetTradingAnalyzer:
    @staticmethod
    def calculate_metrics(transactions, initial_capital):
        """Portfolio-level performance stats"""
        asset_metrics = {}
        if not transactions:
            return {
                "overall": {"total_profit": 0, "overall_return_pct": 0, "total_trades": 0},
                "assets": asset_metrics
            }
            
        for name in set(t["asset"] for t in transactions):
            asset_trans = [t for t in transactions if t["asset"] == name]
            total_profit = sum(t["profit"] for t in asset_trans if t["action"] == "sell")
            total_investment = sum(t["price"] * t["shares"] for t in asset_trans if t["action"] == "buy")
            total_return = (total_profit / total_investment) * 100 if total_investment > 0 else 0
            sell_trans = [t for t in asset_trans if t["action"] == "sell"]
            winning_trades = sum(1 for t in sell_trans if t["profit"] > 0)
            win_rate = (winning_trades / len(sell_trans)) * 100 if sell_trans else 0
            asset_metrics[name] = {
                "total_profit": total_profit,
                "total_investment": total_investment,
                "total_return_pct": total_return,
                "total_trades": len(sell_trans),
                "win_rate": win_rate
            }
        
        total_profit_all = sum(m["total_profit"] for m in asset_metrics.values())
        final_capital = initial_capital + total_profit_all
        overall_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        return {
            "overall": {
                "total_profit": total_profit_all,
                "overall_return_pct": overall_return,
                "total_trades": sum(m["total_trades"] for m in asset_metrics.values())
            },
            "assets": asset_metrics
        }
    
    @staticmethod
    def visualize(env, title, save_path):
        """Plot multi-asset backtest"""
        n_assets = len(env.asset_names)
        fig, axes = plt.subplots(n_assets, 1, figsize=(16, 6 * n_assets), sharex=True)
        if n_assets == 1:
            axes = [axes]
        
        for i, name in enumerate(env.asset_names):
            ax = axes[i]
            prices = env.prices[name]
            dates = env.dates
            ax.plot(dates, prices, label=f'{name} Price', color='blue', alpha=0.6)
            
            buy_signals = [t for t in env.transactions if t["asset"] == name and t["action"] == "buy"]
            sell_signals = [t for t in env.transactions if t["asset"] == name and t["action"] == "sell"]
            
            ax.scatter(
                [t["date"] for t in buy_signals],
                [t["price"] for t in buy_signals],
                marker='^', color='green', label='Buy', s=100, zorder=3
            )
            ax.scatter(
                [t["date"] for t in sell_signals],
                [t["price"] for t in sell_signals],
                marker='v', color='red', label='Sell', s=100, zorder=3
            )
            
            in_position = False
            start_date = None
            for t in env.transactions:
                if t["asset"] == name:
                    if t["action"] == "buy":
                        in_position = True
                        start_date = t["date"]
                    elif t["action"] == "sell" and in_position:
                        ax.axvspan(start_date, t["date"], color='gray', alpha=0.2)
                        in_position = False
            
            ax.set_title(f'{name} Trading Signals', fontsize=12)
            ax.set_ylabel('Price', fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)
        
        metrics = MultiAssetTradingAnalyzer.calculate_metrics(env.transactions, env.initial_capital)
        plt.suptitle(
            f"{title}\nPortfolio Total Profit: {metrics['overall']['total_profit']:.2f} | "
            f"Total Return Rate: {metrics['overall']['overall_return_pct']:.2f}% | "
            f"Total Trades: {metrics['overall']['total_trades']}",
            fontsize=14, y=1.02
        )
        
        plt.xlabel('Date', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# ----------------------------
# Model comparison
# ----------------------------
def compare_models(model_results, output_folder):
    """Compare models and write summary plots"""
    metrics_data = []
    for model_name, result in model_results.items():
        if 'strategy_metrics' in result:
            metrics = result['strategy_metrics']
            metrics_data.append({
                'Model': model_name,
                'Return_Rate_mean': metrics['Tr'],
                'Sharpe_Ratio_mean': metrics['Sr'],
                'Volatility_mean': metrics['Vol'],
                'Max_Drawdown_mean': metrics['Mdd']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    if output_folder:
        metrics_df.to_csv(os.path.join(output_folder, 'model_comparison_metrics.csv'), index=False)
    
    plt.figure(figsize=(10, 8))
    
    labels = ['Return_Rate_mean', 'Sharpe_Ratio_mean', 'Volatility_mean', 'Max_Drawdown_mean']
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for i, row in metrics_df.iterrows():
        values = row[labels].tolist()
        values[2] = max(0, 100 - values[2])
        values[3] = max(0, 100 + values[3])
        values += values[:1]
        
        plt.polar(angles, values, label=row['Model'], linewidth=2, linestyle='solid', marker='o')
        plt.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], labels)
    plt.ylim(0, 100)
    plt.title('Model Performance Comparison', size=20, color='navy', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'model_comparison_radar.png'))
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16)
    
    metrics_to_plot = [
        ('Return_Rate_mean', 'Return Rate (%)', 'green'),
        ('Sharpe_Ratio_mean', 'Sharpe Ratio', 'blue'),
        ('Volatility_mean', 'Volatility (%)', 'orange'),
        ('Max_Drawdown_mean', 'Max Drawdown (%)', 'red')
    ]
    
    for i, (metric, title, color) in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]
        ax.bar(metrics_df['Model'], metrics_df[metric], color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_xticklabels(metrics_df['Model'], rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_folder:
        plt.savefig(os.path.join(output_folder, 'model_comparison_bars.png'))
    plt.close()
    
    return metrics_df


# ----------------------------
# Data loading
# ----------------------------
def load_local_financial_data(csv_path):
    """Load local OHLCV CSV"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path}: {len(df)} rows")
        
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        
        df.columns = [col.lower() for col in df.columns]
        
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        raise


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Compare multiple trading strategies including A2C, LSTM, CNN, ANN and ARIMA')

    parser.add_argument('--train-start', type=str, default="2010-01-03", 
                        help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', type=str, default="2019-12-30", 
                        help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', type=str, default="2020-01-03", 
                        help='Testing start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', type=str, default="2020-12-30", 
                        help='Testing end date (YYYY-MM-DD)')
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH, 
                        help='Path to local financial data CSV file')
    parser.add_argument('--output-dir', default='./model_comparison_results', type=str, 
                        help='Directory to save output results')
    parser.add_argument('--total-timesteps', type=int, default=10000, 
                        help='Total training timesteps for each model')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--buy-ratio', type=float, default=0.2, 
                        help='Ratio of capital to use for each buy (0.0-1.0)')
    parser.add_argument('--sell-ratio', type=float, default=0.5, 
                        help='Ratio of holdings to sell each time (0.0-1.0)')

    args = parser.parse_args()

    set_seed(args.seed)
    
    try:
        TRAIN_START = pd.Timestamp(args.train_start)
        TRAIN_END = pd.Timestamp(args.train_end)
        TEST_START = pd.Timestamp(args.test_start)
        TEST_END = pd.Timestamp(args.test_end)
    except ValueError as e:
        print(f"Invalid date: {e}")
        print("Use YYYY-MM-DD dates")
        return
    
    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # ==============================================
    # ==============================================
    hyperparams = {
        'seed': args.seed,
        
        'window_size': 20,
        'initial_capital': 10000,
        'trade_fee': 0.0005,
        'hold_penalty_threshold': 4,
        'hold_penalty_factor': 1.0,
        'hold_severe_threshold': 10,
        'hold_severe_penalty': 10.0,
        'buy_reward': 2.0,
        'insufficient_fund_penalty': 1.0,
        'profit_reward_factor': 0.5,
        'value_increase_factor': 0.1,
        'invalid_action_penalty': 0.1,
        'train_dates': (TRAIN_START, TRAIN_END),
        
        'buy_ratio': args.buy_ratio,
        'sell_ratio': args.sell_ratio,
        
        'a2c_hidden_dim': 128,
        'a2c_lr': 0.00001,
        'a2c_gamma': 0.96,
        'a2c_entropy_coef': 0.05,
        'a2c_value_coef': 0.5,
        'a2c_rollout_steps': 50,
        'a2c_total_timesteps': args.total_timesteps,
        
        'lstm_hidden_dim': 128,
        'lstm_num_layers': 2,
        'lstm_lr': 0.0001,
        'lstm_batch_size': 32,
        'lstm_episode_length': 100,
        'lstm_total_timesteps': args.total_timesteps,
        
        'cnn_hidden_dim': 128,
        'cnn_lr': 0.0001,
        'cnn_episode_length': 100,
        'cnn_total_timesteps': args.total_timesteps,
        
        'ann_hidden_dim': 256,
        'ann_lr': 0.0001,
        'ann_episode_length': 100,
        'ann_total_timesteps': args.total_timesteps,
        
        'arima_order': (5, 1, 0),
        'arima_threshold': 0.005
    }
    
    with open(os.path.join(output_folder, f'hyperparameters_{args.total_timesteps//1000}k.log'), 'w') as f:
        for key, value in hyperparams.items():
            f.write(f"{key}: {value}\n")
    print("Hyperparameters logged to hyperparameters.log")


    # ==============================================
    # ==============================================
    try:
        df = load_local_financial_data(args.data_path)
        print(f"Dataset span: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Dataset load failed: {e}")
        return


    target_symbols = {
        "AAPL": "Apple Inc.",
        "AXP": "American Express",
        "BAC": "Bank of America",
        "CCL": "Carnival Corporation",
        "CVX": "Chevron Corporation",
        "JNJ": "Johnson & Johnson",
        "MRO": "Marathon Oil",
        "MSFT": "Microsoft Corporation",
        "NVDA": "NVIDIA Corporation",
        "OXY": "Occidental Petroleum",
        "RCL": "Royal Caribbean Cruises"
    }


    filtered_df = df[df['symbol'].isin(target_symbols.keys())].copy()
    filtered_df['symbol_fullname'] = filtered_df['symbol'].map(target_symbols)


    if len(filtered_df) == 0:
        print("No target symbols found; check ticker list")
        return


    asset_dfs = {}
    for symbol in target_symbols.keys():
        stock_all_df = filtered_df[filtered_df['symbol'] == symbol].copy()
        stock_all_df = stock_all_df.sort_values('date').set_index('date')
        
        train_mask = (stock_all_df.index >= TRAIN_START) & (stock_all_df.index <= TRAIN_END)
        test_mask = (stock_all_df.index >= TEST_START) & (stock_all_df.index <= TEST_END)
        
        has_train_data = len(stock_all_df[train_mask]) >= hyperparams['window_size']
        has_test_data = len(stock_all_df[test_mask]) >= hyperparams['window_size']
        
        if has_train_data and has_test_data:
            asset_dfs[symbol] = stock_all_df
            print(f"\nLoaded {symbol} ({target_symbols[symbol]}): {len(stock_all_df)} rows, {stock_all_df.index[0].strftime('%Y-%m-%d')} to {stock_all_df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"\nWarning: skipping {symbol} (insufficient train/test coverage)")


    if not asset_dfs:
        print("No valid assets; exiting.")
        return


    train_dates = (TRAIN_START, TRAIN_END)
    test_dates = (TEST_START, TEST_END)


    # ==============================================
    # ==============================================
    model_results = {}
    
    print("\n===== Random baseline =====")
    random_metrics = run_random_strategy(
        asset_dfs,
        window_size=hyperparams['window_size'],
        test_dates=test_dates,
        hyperparams=hyperparams,
        output_folder=output_folder
    )
    model_results['Random'] = random_metrics
    
    print("\n===== Training A2C =====")
    a2c_metrics, _ = train_a2c(
        asset_dfs,
        window_size=hyperparams['window_size'],
        train_dates=train_dates,
        test_dates=test_dates,
        hyperparams=hyperparams,
        output_folder=output_folder,
        model_filename_base='a2c_model_for_comparison'
    )
    model_results['A2C'] = a2c_metrics
    
    print("\n===== Training LSTM =====")
    lstm_metrics, _ = train_lstm(
        asset_dfs,
        window_size=hyperparams['window_size'],
        train_dates=train_dates,
        test_dates=test_dates,
        hyperparams=hyperparams,
        output_folder=output_folder
    )
    model_results['LSTM'] = lstm_metrics
    
    print("\n===== Training CNN =====")
    cnn_metrics, _ = train_cnn(
        asset_dfs,
        window_size=hyperparams['window_size'],
        train_dates=train_dates,
        test_dates=test_dates,
        hyperparams=hyperparams,
        output_folder=output_folder
    )
    model_results['CNN'] = cnn_metrics
    
    print("\n===== Training MLP =====")
    ann_metrics, _ = train_ann(
        asset_dfs,
        window_size=hyperparams['window_size'],
        train_dates=train_dates,
        test_dates=test_dates,
        hyperparams=hyperparams,
        output_folder=output_folder
    )
    model_results['ANN'] = ann_metrics
    
    print("\n===== Running ARIMA =====")
    arima_metrics = run_arima_strategy(
        asset_dfs,
        window_size=hyperparams['window_size'],
        test_dates=test_dates,
        hyperparams=hyperparams,
        output_folder=output_folder
    )
    model_results['ARIMA'] = arima_metrics
    
    # ==============================================
    # ==============================================
    print("\n===== Model comparison =====")
    comparison_df = compare_models(model_results, output_folder)
    print(comparison_df.to_string(index=False))
    
    print(f"\nAll results saved to {output_folder}")


if __name__ == "__main__":
    main()
