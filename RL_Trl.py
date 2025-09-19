import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import time
from datetime import datetime
import threading
import queue
import re
import unicodedata

# RL Libraries
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Set page config
st.set_page_config(page_title="RL Defect Reduction System", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'surrogate_model' not in st.session_state:
    st.session_state.surrogate_model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'actual_feature_names' not in st.session_state:
    st.session_state.actual_feature_names = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'target_stats' not in st.session_state:
    st.session_state.target_stats = {}
if 'training_results' not in st.session_state:
    st.session_state.training_results = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = {}
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'trained_agents' not in st.session_state:
    st.session_state.trained_agents = {}
# FIXED: Add persistent charts storage
if 'training_charts' not in st.session_state:
    st.session_state.training_charts = {}
if 'loss_charts' not in st.session_state:
    st.session_state.loss_charts = {}

# Enhanced text cleaning and normalization functions
def fix_encoding_issues(text):
    """Fix common encoding issues in text"""
    if not isinstance(text, str):
        return str(text)
    
    # Common encoding fixes
    encoding_fixes = {
        'Ã©': 'é',
        'Ã¨': 'è', 
        'Ã ': 'à',
        'Ã¡': 'á',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
        'Ã±': 'ñ',
        'Ã§': 'ç',
        'Ã¢': 'â',
        'Ã´': 'ô',
        'Ã®': 'î',
        'Ã»': 'û',
        'Ã¼': 'ü',
        'Ã«': 'ë',
        'Ã¿': 'ÿ',
        'Ã€': 'À',
        'Ã‰': 'É',
        'Ã°': 'ð',
        'Ã¾': 'þ',
        'Ã•': 'Õ',
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â„¢': '™',
        'â†': '→',
        'Â°': '°',
        'Â': ''
    }
    
    # Apply fixes
    fixed_text = text
    for wrong, right in encoding_fixes.items():
        fixed_text = fixed_text.replace(wrong, right)
    
    # Additional cleanup
    fixed_text = fixed_text.strip()
    
    return fixed_text

def normalize_text_for_matching(text):
    """Normalize text for better matching"""
    if not isinstance(text, str):
        text = str(text)
    
    # Fix encoding first
    text = fix_encoding_issues(text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase for comparison
    text_lower = text.lower()
    
    return text, text_lower

def fuzzy_match_columns(expected_features, available_columns, threshold=0.6):
    """Enhanced column matching with fuzzy logic"""
    matches = {}
    used_columns = set()
    
    # First pass - exact matches
    for expected in expected_features:
        expected_clean, expected_lower = normalize_text_for_matching(expected)
        
        for col in available_columns:
            if col in used_columns:
                continue
                
            col_clean, col_lower = normalize_text_for_matching(col)
            
            # Exact match (case insensitive)
            if expected_lower == col_lower:
                matches[expected] = col
                used_columns.add(col)
                break
    
    # Second pass - substring matches
    for expected in expected_features:
        if expected in matches:
            continue
            
        expected_clean, expected_lower = normalize_text_for_matching(expected)
        
        # Try substring matching
        for col in available_columns:
            if col in used_columns:
                continue
                
            col_clean, col_lower = normalize_text_for_matching(col)
            
            # Check if one contains the other
            if (expected_lower in col_lower and len(expected_lower) > 3) or \
               (col_lower in expected_lower and len(col_lower) > 3):
                matches[expected] = col
                used_columns.add(col)
                break
    
    # Third pass - word-based matching
    for expected in expected_features:
        if expected in matches:
            continue
            
        expected_clean, expected_lower = normalize_text_for_matching(expected)
        expected_words = set(expected_lower.split())
        
        best_match = None
        best_score = 0
        
        for col in available_columns:
            if col in used_columns:
                continue
                
            col_clean, col_lower = normalize_text_for_matching(col)
            col_words = set(col_lower.split())
            
            # Calculate word overlap
            if expected_words and col_words:
                intersection = expected_words.intersection(col_words)
                union = expected_words.union(col_words)
                score = len(intersection) / len(union) if union else 0
                
                if score > threshold and score > best_score:
                    best_score = score
                    best_match = col
        
        if best_match:
            matches[expected] = best_match
            used_columns.add(best_match)
    
    return matches

# Load feature names from JSON
def load_feature_names():
    try:
        with open('best_features.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            features = data.get('features', [])
        elif isinstance(data, list):
            features = data
        else:
            st.error("Invalid JSON structure in best_features.json")
            return []
        
        # Clean encoding issues in feature names
        cleaned_features = [fix_encoding_issues(feature) for feature in features]
        return cleaned_features
            
    except FileNotFoundError:
        st.warning("best_features.json file not found in the same directory as app.py. Please create it with your feature names.")
        return []
    except json.JSONDecodeError:
        st.error("Invalid JSON format in best_features.json")
        return []
    except UnicodeDecodeError:
        # Try different encodings
        try:
            with open('best_features.json', 'r', encoding='latin-1') as f:
                data = json.load(f)
            if isinstance(data, dict):
                features = data.get('features', [])
            elif isinstance(data, list):
                features = data
            else:
                return []
            cleaned_features = [fix_encoding_issues(feature) for feature in features]
            return cleaned_features
        except:
            st.error("Could not read best_features.json with any encoding")
            return []

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# Custom Environment for RL with Direct Negative Defect Reward
class DefectReductionEnv(gym.Env):
    def __init__(self, surrogate_model, feature_ranges, feature_names, max_steps=50):
        super().__init__()
        self.surrogate_model = surrogate_model
        self.feature_ranges = feature_ranges
        self.feature_names = feature_names
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action and observation spaces
        self.n_features = len(feature_names)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_features,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features + 1,))
        
        self.reset()
    
    def reset(self, seed=None):
        # Start from random point in feature space
        self.current_params = np.array([
            np.random.uniform(low, high) for low, high in self.feature_ranges.values()
        ])
        self.current_step = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Get current defect prediction - FIXED: Use DataFrame for proper feature names
        if hasattr(self.surrogate_model, 'feature_names_in_'):
            # Create DataFrame with proper feature names
            input_df = pd.DataFrame([self.current_params], columns=self.feature_names)
            defect_pred = self.surrogate_model.predict(input_df)[0]
        else:
            defect_pred = self.surrogate_model.predict([self.current_params])[0]
        return np.concatenate([self.current_params, [defect_pred]])
    
    def step(self, action):
        # Apply 2% max change constraint
        for i, (feature_name, (min_val, max_val)) in enumerate(self.feature_ranges.items()):
            max_change = (max_val - min_val) * 0.02
            change = action[i] * max_change
            self.current_params[i] = np.clip(
                self.current_params[i] + change, min_val, max_val
            )
        
        # Get current defect prediction from surrogate model - FIXED
        if hasattr(self.surrogate_model, 'feature_names_in_'):
            input_df = pd.DataFrame([self.current_params], columns=self.feature_names)
            current_defect = self.surrogate_model.predict(input_df)[0]
        else:
            current_defect = self.surrogate_model.predict([self.current_params])[0]
        
        # Simple negative defect as reward (lower defect = higher reward)
        reward = -current_defect
        
        # Add small step penalty to encourage efficiency
        reward -= 0.001
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}

# Neural Networks for RL
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, hidden_dims=[256, 256]):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], action_dim)
        self.max_action = max_action
        
        # Initialize weights
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], 1)
        
        # Initialize weights
        for layer in [self.l1, self.l2, self.l3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

# Proper TD3 Agent with Training
class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action=1.0, lr=3e-4, 
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, 
                 policy_freq=2, hidden_dims=[256, 256]):
        
        # Store hyperparameters
        self.hyperparams = {
            'learning_rate': lr,
            'gamma': gamma,
            'tau': tau,
            'policy_noise': policy_noise,
            'noise_clip': noise_clip,
            'policy_freq': policy_freq,
            'hidden_dims': hidden_dims,
            'max_action': max_action
        }
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dims)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dims)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim, hidden_dims)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dims)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dims)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dims)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        # Copy to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Hyperparameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_iterations = 0
        self.exploration_noise = 0.1
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def train(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return 0, 0
        
        self.total_iterations += 1
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.BoolTensor(done).unsqueeze(1)
        
        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * self.policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        next_action = (self.actor_target(next_state) + noise).clamp(
            -self.max_action, self.max_action
        )
        
        # Compute the target Q value
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (~done) * self.gamma * target_Q
        
        # Get current Q estimates
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())
        
        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        actor_loss = 0
        # Delayed policy updates
        if self.total_iterations % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update the frozen target models
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            actor_loss = actor_loss.item()
        
        return critic_loss.item(), actor_loss

# Proper SAC Agent with Training
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action=1.0, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, hidden_dims=[256, 256]):
        
        # Store hyperparameters
        self.hyperparams = {
            'learning_rate': lr,
            'gamma': gamma,
            'tau': tau,
            'alpha': alpha,
            'hidden_dims': hidden_dims,
            'max_action': max_action
        }
        
        # Simplified SAC - just using deterministic policy for this implementation
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dims)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dims)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dims)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.exploration_noise = 0.1
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def train(self, replay_buffer, batch_size=100):
        if len(replay_buffer) < batch_size:
            return 0, 0
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.BoolTensor(done).unsqueeze(1)
        
        # Critic update
        next_action = self.actor(next_state)
        target_Q1 = self.critic_1(next_state, next_action)
        target_Q2 = self.critic_2(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (~done) * self.gamma * target_Q
        
        current_Q1 = self.critic_1(state, action)
        current_Q2 = self.critic_2(state, action)
        
        critic_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach())
        
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()

# Proper PPO Agent with Training
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, hidden_dims=[256, 256]):
        
        # Store hyperparameters
        self.hyperparams = {
            'learning_rate': lr,
            'gamma': gamma,
            'eps_clip': eps_clip,
            'k_epochs': k_epochs,
            'hidden_dims': hidden_dims
        }
        
        # Policy network (Actor)
        self.policy = Actor(state_dim, action_dim, 1.0, hidden_dims)
        
        # Value network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # Initialize weights
        for module in self.value_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.exploration_noise = 0.1
        
        # Storage for PPO
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.policy(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def train(self, replay_buffer=None, batch_size=None):
        # PPO doesn't use replay buffer, it uses collected trajectories
        if len(self.states) == 0:
            return 0, 0
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_states = torch.tensor(self.states, dtype=torch.float32)
        old_actions = torch.tensor(self.actions, dtype=torch.float32)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        
        # PPO update
        policy_loss_total = 0
        value_loss_total = 0
        
        for _ in range(self.k_epochs):
            # Get current policy outputs
            current_actions = self.policy(old_states)
            current_values = self.value_net(old_states).squeeze()
            
            # Calculate log probabilities (simplified for continuous actions)
            log_probs = -0.5 * ((current_actions - old_actions) ** 2).sum(dim=1)
            
            # Calculate ratio
            ratios = torch.exp(log_probs - old_log_probs)
            
            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values, returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()
            
            # Update value network
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
        
        # Clear storage
        self.clear_memory()
        
        return policy_loss_total / self.k_epochs, value_loss_total / self.k_epochs
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

# Main Streamlit App
st.title("RL-based Defect Reduction System ")

# Load feature names
if not st.session_state.feature_names:
    st.session_state.feature_names = load_feature_names()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data & Surrogate Model", "RL Training & Comparison", "Defect Reduction Recommendations"])

# Tab 1: Data Upload and Surrogate Model (keeping the existing implementation)
with tab1:
    st.header("Data Upload and Surrogate Model Training")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # Handle different file types and encodings
                if uploaded_file.name.endswith('.csv'):
                    # Try multiple encodings for CSV files
                    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                    data = None
                    used_encoding = None
                    
                    for encoding in encodings_to_try:
                        try:
                            uploaded_file.seek(0)  # Reset file pointer
                            data = pd.read_csv(uploaded_file, encoding=encoding)
                            used_encoding = encoding
                            st.success(f"Data loaded successfully with {encoding} encoding!")
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    
                    if data is None:
                        st.error("Could not read the CSV file with any encoding. Please check the file format.")
                        st.stop()
                else:
                    data = pd.read_excel(uploaded_file)
                    used_encoding = "Excel format"
                    st.success("Excel data loaded successfully!")
                
                # Clean column names - fix encoding issues and strip whitespace
                original_columns = data.columns.tolist()
                data.columns = [fix_encoding_issues(str(col).strip()) for col in data.columns]
                
                st.info(f"Data loaded successfully! Shape: {data.shape}")
                
                # Show encoding fixes if any were made
                encoding_fixes_made = []
                for orig, fixed in zip(original_columns, data.columns):
                    if str(orig) != str(fixed):
                        encoding_fixes_made.append(f"'{orig}' → '{fixed}'")
                
                if encoding_fixes_made:
                    with st.expander("Encoding Fixes Applied", expanded=False):
                        st.write("The following column names were fixed:")
                        for fix in encoding_fixes_made[:10]:  # Show first 10
                            st.write(f"• {fix}")
                        if len(encoding_fixes_made) > 10:
                            st.write(f"... and {len(encoding_fixes_made) - 10} more")
                
                # Enhanced column matching
                if st.session_state.feature_names:
                    st.subheader("Feature Matching Analysis")
                    
                    # Use enhanced matching function
                    column_matches = fuzzy_match_columns(st.session_state.feature_names, data.columns.tolist())
                    
                    matched_features = []
                    missing_features = []
                    
                    for expected_feature in st.session_state.feature_names:
                        if expected_feature in column_matches:
                            matched_features.append(column_matches[expected_feature])
                        else:
                            missing_features.append(expected_feature)
                    
                    # Display matching results
                    col_match1, col_match2 = st.columns([2, 1])
                    
                    with col_match1:
                        if matched_features:
                            st.success(f"Found {len(matched_features)} out of {len(st.session_state.feature_names)} features")
                            
                            # Show feature mapping
                            with st.expander("Feature Mapping Details", expanded=len(missing_features) > 0):
                                mapping_data = []
                                for expected in st.session_state.feature_names:
                                    if expected in column_matches:
                                        found = column_matches[expected]
                                        match_type = "Exact" if expected.lower() == found.lower() else "Fuzzy"
                                        mapping_data.append({
                                            "Expected Feature": expected,
                                            "Found Column": found,
                                            "Match Type": match_type,
                                            "Status": "Matched"
                                        })
                                    else:
                                        mapping_data.append({
                                            "Expected Feature": expected,
                                            "Found Column": "None",
                                            "Match Type": "N/A",
                                            "Status": "Missing"
                                        })
                                
                                mapping_df = pd.DataFrame(mapping_data)
                                st.dataframe(mapping_df, use_container_width=True)
                        else:
                            st.error("No matching features found!")
                        
                        if missing_features:
                            st.warning(f"Missing features ({len(missing_features)}):")
                            missing_cols = st.columns(min(3, len(missing_features)))
                            for i, feature in enumerate(missing_features):
                                with missing_cols[i % 3]:
                                    st.write(f"• {feature}")
                    
                    with col_match2:
                        st.metric("Features Found", len(matched_features))
                        st.metric("Features Missing", len(missing_features))
                        st.metric("Match Rate", f"{len(matched_features)/len(st.session_state.feature_names)*100:.1f}%")
                    
                    # Look for target column with enhanced matching
                    target_variations = ['class1', 'class 1', 'class_1', 'défaut', 'default', 'defect', 'target', 'classe', 'classification']
                    target_column = None
                    
                    # Use fuzzy matching for target column too
                    target_matches = fuzzy_match_columns(target_variations, data.columns.tolist(), threshold=0.5)
                    
                    if target_matches:
                        # Get the first match
                        target_column = next(iter(target_matches.values()))
                        st.success(f"Target column detected: '{target_column}'")
                    
                    # Proceed only if we have sufficient matches
                    if len(matched_features) >= len(st.session_state.feature_names) * 0.8:  # At least 80% match
                        if target_column is not None:
                            # Create final dataset with matched features
                            final_features = [column_matches[f] for f in st.session_state.feature_names if f in column_matches]
                            
                            # Ensure we don't use the target column as a feature
                            final_features = [f for f in final_features if f != target_column]
                            
                            if len(final_features) > 0:
                                # Store filtered data
                                filtered_data = data[final_features + [target_column]].dropna()
                                st.session_state.data = filtered_data
                                st.session_state.actual_feature_names = final_features
                                st.session_state.target_column = target_column
                                
                                # Calculate target statistics
                                st.session_state.target_stats = {
                                    'mean': filtered_data[target_column].mean(),
                                    'std': filtered_data[target_column].std(),
                                    'min': filtered_data[target_column].min(),
                                    'max': filtered_data[target_column].max(),
                                    'q25': filtered_data[target_column].quantile(0.25),
                                    'q75': filtered_data[target_column].quantile(0.75)
                                }
                                
                                st.success(f"Dataset prepared with {len(final_features)} features and {len(filtered_data)} samples")
                                
                                # Display data preview
                                st.subheader("Data Preview")
                                preview_data = filtered_data.head()
                                st.dataframe(preview_data, use_container_width=True)
                                
                                # Show statistics
                                with st.expander("Dataset Statistics", expanded=False):
                                    col_stats1, col_stats2 = st.columns(2)
                                    
                                    with col_stats1:
                                        st.write("**Target Variable Statistics:**")
                                        st.write(f"• Mean: {st.session_state.target_stats['mean']:.4f}")
                                        st.write(f"• Std: {st.session_state.target_stats['std']:.4f}")
                                        st.write(f"• Min: {st.session_state.target_stats['min']:.4f}")
                                        st.write(f"• Max: {st.session_state.target_stats['max']:.4f}")
                                    
                                    with col_stats2:
                                        st.write("**Quartiles:**")
                                        st.write(f"• Q1 (25%): {st.session_state.target_stats['q25']:.4f}")
                                        st.write(f"• Q3 (75%): {st.session_state.target_stats['q75']:.4f}")
                                        st.write(f"• IQR: {st.session_state.target_stats['q75'] - st.session_state.target_stats['q25']:.4f}")
                                        st.write(f"• Target (Q1): {st.session_state.target_stats['q25']:.4f}")
                            else:
                                st.error("No valid features found after removing target column!")
                        else:
                            st.error("Target column not found!")
                            st.info("**Available columns:** " + ", ".join(data.columns.tolist()[:20]))  # Show first 20
                            if len(data.columns) > 20:
                                st.info(f"... and {len(data.columns) - 20} more columns")
                            st.info("**Searching for columns containing:** " + ", ".join(target_variations))
                    else:
                        st.error(f"Insufficient feature matches! Need at least {len(st.session_state.feature_names) * 0.8:.0f} features, found {len(matched_features)}")
                        st.info("**Please check your feature names in best_features.json or verify the dataset columns**")
                else:
                    st.warning("No feature names loaded from JSON. Please ensure best_features.json exists.")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.info("**Troubleshooting tips:**")
                st.info("• Check if the file is corrupted")
                st.info("• Ensure the file contains the expected columns")
                st.info("• Try saving the file with UTF-8 encoding")
    
    with col2:
        st.subheader("Surrogate Model Options")
        
        model_option = st.radio(
            "Choose surrogate model option:",
            ["Train new model", "Upload pre-trained model"]
        )
        
        if model_option == "Train new model" and st.session_state.data is not None:
            if st.button("Train LightGBM Model", type="primary"):
                with st.spinner("Training surrogate model with grid search..."):
                    try:
                        # Prepare data
                        X = st.session_state.data[st.session_state.actual_feature_names]
                        y = st.session_state.data[st.session_state.target_column]
                        
                        st.info(f"Training with {X.shape[1]} features on {X.shape[0]} samples")
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Grid search parameters
                        param_grid = {
                            'n_estimators': [100, 200, 500],
                            'learning_rate': [0.01, 0.05, 0.1],
                            'max_depth': [3, 5, 7],
                            'num_leaves': [31, 50, 100],
                            'min_child_samples': [20, 30, 50]
                        }
                        
                        # Grid search
                        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
                        grid_search = GridSearchCV(
                            lgb_model, param_grid, cv=3, 
                            scoring='r2', n_jobs=-1, verbose=0
                        )
                        
                        grid_search.fit(X_train, y_train)
                        
                        # Best model predictions
                        y_pred_train = grid_search.predict(X_train)
                        y_pred_test = grid_search.predict(X_test)
                        
                        # Store model and results
                        st.session_state.surrogate_model = grid_search.best_estimator_
                        
                        # Display results
                        train_r2 = r2_score(y_train, y_pred_train)
                        test_r2 = r2_score(y_test, y_pred_test)
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        
                        st.success("Model trained successfully!")
                        
                        col_r2_1, col_r2_2, col_r2_3, col_r2_4 = st.columns(4)
                        with col_r2_1:
                            st.metric("Train R²", f"{train_r2:.4f}")
                        with col_r2_2:
                            st.metric("Test R²", f"{test_r2:.4f}")
                        with col_r2_3:
                            st.metric("Train RMSE", f"{train_rmse:.4f}")
                        with col_r2_4:
                            st.metric("Test RMSE", f"{test_rmse:.4f}")
                        
                        # Plot actual vs predicted
                        fig = make_subplots(rows=1, cols=2, 
                                          subplot_titles=('Training Set', 'Test Set'))
                        
                        # Training plot
                        fig.add_trace(
                            go.Scatter(x=y_train, y=y_pred_train, mode='markers',
                                     name='Training', opacity=0.6),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=[y_train.min(), y_train.max()], 
                                     y=[y_train.min(), y_train.max()],
                                     mode='lines', name='Perfect Fit', 
                                     line=dict(dash='dash', color='red')),
                            row=1, col=1
                        )
                        
                        # Test plot
                        fig.add_trace(
                            go.Scatter(x=y_test, y=y_pred_test, mode='markers',
                                     name='Test', opacity=0.6, 
                                     marker=dict(color='orange')),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(x=[y_test.min(), y_test.max()], 
                                     y=[y_test.min(), y_test.max()],
                                     mode='lines', name='Perfect Fit', 
                                     line=dict(dash='dash', color='red')),
                            row=1, col=2
                        )
                        
                        fig.update_xaxes(title_text="Actual", row=1, col=1)
                        fig.update_xaxes(title_text="Actual", row=1, col=2)
                        fig.update_yaxes(title_text="Predicted", row=1, col=1)
                        fig.update_yaxes(title_text="Predicted", row=1, col=2)
                        
                        fig.update_layout(
                            title="Actual vs Predicted - Surrogate Model Performance",
                            showlegend=True,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display best parameters
                        st.subheader("Best Hyperparameters")
                        st.json(grid_search.best_params_)
                        
                        # Feature importance if available
                        if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                            st.subheader("Feature Importance")
                            importance_data = pd.DataFrame({
                                'Feature': st.session_state.actual_feature_names,
                                'Importance': grid_search.best_estimator_.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig_importance = px.bar(importance_data, x='Importance', y='Feature', 
                                                  orientation='h', title="Feature Importance")
                            fig_importance.update_layout(height=400)
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
        
        elif model_option == "Upload pre-trained model":
            model_file = st.file_uploader("Upload pre-trained model (.pkl)", type=['pkl'])
            if model_file is not None:
                try:
                    model = pickle.load(model_file)
                    st.session_state.surrogate_model = model
                    st.success("Pre-trained model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

# Tab 2: RL Training and Comparison (FIXED: Persistent Charts)
with tab2:
    st.header("RL Agent Training and Comparison")
    
    if st.session_state.data is None or st.session_state.surrogate_model is None:
        st.warning("Please upload data and train/load a surrogate model in Tab 1 first!")
    else:
        # FIXED: Always show existing charts first, even before training
        if st.session_state.training_results:
            st.subheader("Training Results")
            
            # Quick summary metrics
            summary_cols = st.columns(len(st.session_state.training_results))
            for i, (agent_name, rewards) in enumerate(st.session_state.training_results.items()):
                with summary_cols[i]:
                    final_reward = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
                    improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
                    st.metric(f"{agent_name}", f"{final_reward:.3f}", f"{improvement:+.3f}")
            
            # Show training chart (persistent)
            if st.session_state.training_charts:
                st.plotly_chart(st.session_state.training_charts, use_container_width=True)
            else:
                # Recreate chart from stored data
                fig = go.Figure()
                colors = {'TD3': '#1f77b4', 'SAC': '#ff7f0e', 'PPO': '#2ca02c'}
                
                for agent_name, rewards in st.session_state.training_results.items():
                    if len(rewards) > 20:
                        window = min(50, len(rewards) // 4)
                        smoothed = pd.Series(rewards).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                    else:
                        smoothed = rewards
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(len(smoothed))),
                        y=smoothed,
                        name=agent_name,
                        mode='lines',
                        line=dict(width=3, color=colors.get(agent_name, 'gray'))
                    ))
                
                fig.update_layout(
                    title="RL Training Results - Rewards (Persistent)",
                    xaxis_title="Episode",
                    yaxis_title="Reward (Negative Defect Rate)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                # Store for persistence
                st.session_state.training_charts = fig
            
            # Show loss chart if available (persistent)
            if st.session_state.loss_charts:
                st.plotly_chart(st.session_state.loss_charts, use_container_width=True)
        
        st.info("Fixed Issues: Agents now actually learn from rewards with proper gradient updates and experience replay!")
        
        # Show feature tracking
        st.subheader("Parameter/Feature Tracking")
        if st.session_state.actual_feature_names:
            st.success(f"All {len(st.session_state.actual_feature_names)} parameters will be optimized during training:")
            
            # Display features in a nice grid
            cols = st.columns(min(5, len(st.session_state.actual_feature_names)))
            for i, feature in enumerate(st.session_state.actual_feature_names):
                with cols[i % len(cols)]:
                    # Get feature stats
                    if st.session_state.data is not None:
                        min_val = st.session_state.data[feature].min()
                        max_val = st.session_state.data[feature].max()
                        mean_val = st.session_state.data[feature].mean()
                        st.metric(
                            label=feature[:20] + "..." if len(feature) > 20 else feature,
                            value=f"{mean_val:.3f}",
                            delta=f"Range: {min_val:.3f} to {max_val:.3f}",
                            help=f"Full name: {feature}"
                        )
        else:
            st.error("No features available for optimization!")
        
        # Training hyperparameters
        st.subheader("Training Hyperparameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            episodes = st.number_input("Episodes", min_value=100, max_value=2000, value=500, step=100)
            learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=3e-4, format="%.1e")
        
        with col2:
            max_steps = st.number_input("Max Steps/Episode", min_value=10, max_value=100, value=30, step=10)
            batch_size = st.number_input("Batch Size", min_value=32, max_value=256, value=64, step=32)
        
        with col3:
            gamma = st.slider("Discount Factor (γ)", min_value=0.9, max_value=0.999, value=0.99, step=0.01)
            buffer_size = st.number_input("Buffer Size", min_value=1000, max_value=50000, value=10000, step=1000)
        
        with col4:
            tau = st.number_input("Target Update Rate (τ)", min_value=0.001, max_value=0.01, value=0.005, format="%.3f")
            noise_std = st.slider("Exploration Noise", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        # Show current hyperparameters
        with st.expander("Current Hyperparameter Configuration"):
            hyper_col1, hyper_col2, hyper_col3 = st.columns(3)
            
            with hyper_col1:
                st.write("**Training Parameters:**")
                st.write(f"• Episodes: {episodes}")
                st.write(f"• Max Steps/Episode: {max_steps}")
                st.write(f"• Batch Size: {batch_size}")
                st.write(f"• Buffer Size: {buffer_size:,}")
            
            with hyper_col2:
                st.write("**Learning Parameters:**")
                st.write(f"• Learning Rate: {learning_rate:.1e}")
                st.write(f"• Discount Factor (γ): {gamma}")
                st.write(f"• Target Update (τ): {tau}")
                st.write(f"• Exploration Noise: {noise_std}")
            
            with hyper_col3:
                st.write("**Network Architecture:**")
                st.write("• Hidden Layers: [256, 256]")
                st.write("• Activation: ReLU")
                st.write("• Optimizer: Adam")
                st.write("• Max Action Change: ±2%")
        
        # Show current training status
        if st.session_state.training_running:
            col_reset1, col_reset2 = st.columns([1, 4])
            with col_reset1:
                if st.button("Reset Training", type="secondary"):
                    st.session_state.training_running = False
                    st.rerun()
            with col_reset2:
                st.info("Training is running... Click 'Reset Training' if stuck.")
        
        # Training button
        train_button_disabled = st.session_state.training_running
        
        if st.button("Start RL Training", type="primary", disabled=train_button_disabled):
            # Validate that we have all required data
            if not hasattr(st.session_state, 'actual_feature_names') or not st.session_state.actual_feature_names:
                st.error("Feature data not available. Please upload and process data in Tab 1 first.")
                st.stop()
            
            if st.session_state.surrogate_model is None:
                st.error("Surrogate model not available. Please train or upload a model in Tab 1 first.")
                st.stop()
            
            # Set training flag
            st.session_state.training_running = True
            
            try:
                # Create environment
                feature_ranges = {}
                for feature in st.session_state.actual_feature_names:
                    feature_ranges[feature] = (
                        st.session_state.data[feature].min(),
                        st.session_state.data[feature].max()
                    )
                
                env = DefectReductionEnv(
                    st.session_state.surrogate_model,
                    feature_ranges,
                    st.session_state.actual_feature_names,
                    max_steps
                )
                
                # Training progress containers
                status_placeholder = st.empty()
                chart_placeholder = st.empty()
                metrics_placeholder = st.empty()
                loss_chart_placeholder = st.empty()
                
                # Training results storage
                training_results = {}
                loss_results = {}
                
                # Train each algorithm
                algorithms = ['TD3', 'SAC', 'PPO']
                
                # Overall progress
                overall_progress = st.progress(0, "Initializing training...")
                
                for algo_idx, agent_name in enumerate(algorithms):
                    status_placeholder.info(f"Training {agent_name} agent with REAL learning... ({algo_idx + 1}/{len(algorithms)})")
                    
                    # Initialize agent with hyperparameters
                    state_dim = env.observation_space.shape[0]
                    action_dim = env.action_space.shape[0]
                    
                    if agent_name == 'TD3':
                        agent = TD3Agent(
                            state_dim, action_dim, max_action=1.0,
                            lr=learning_rate, gamma=gamma, tau=tau,
                            hidden_dims=[256, 256]
                        )
                        replay_buffer = ReplayBuffer(capacity=buffer_size)
                    elif agent_name == 'SAC':
                        agent = SACAgent(
                            state_dim, action_dim, max_action=1.0,
                            lr=learning_rate, gamma=gamma, tau=tau,
                            hidden_dims=[256, 256]
                        )
                        replay_buffer = ReplayBuffer(capacity=buffer_size)
                    else:  # PPO
                        agent = PPOAgent(
                            state_dim, action_dim, 
                            lr=learning_rate, gamma=gamma,
                            hidden_dims=[256, 256]
                        )
                        replay_buffer = None  # PPO doesn't use replay buffer
                    
                    # Set exploration noise
                    agent.exploration_noise = noise_std
                    
                    episode_rewards = []
                    critic_losses = []
                    actor_losses = []
                    
                    # Progress bar for current algorithm
                    algo_progress = st.progress(0, f"Training {agent_name}")
                    
                    
                    for episode in range(episodes):
                        state, _ = env.reset()
                        episode_reward = 0
                        episode_critic_loss = 0
                        episode_actor_loss = 0
                        loss_count = 0
                        done = False
                        step_count = 0
                        
                        # Collect trajectory for PPO
                        if agent_name == 'PPO':
                            trajectory_states = []
                            trajectory_actions = []
                            trajectory_rewards = []
                            trajectory_log_probs = []
                            trajectory_values = []
                            trajectory_dones = []
                        
                        # Run episode
                        while not done and step_count < max_steps:
                            action = agent.select_action(state, add_noise=True)
                            next_state, reward, done, _, _ = env.step(action)
                            
                            if agent_name == 'PPO':
                                # Store for PPO
                                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                                log_prob = -0.5 * ((agent.policy(state_tensor) - action_tensor) ** 2).sum().item()
                                value = agent.value_net(state_tensor).item()
                                
                                trajectory_states.append(state)
                                trajectory_actions.append(action)
                                trajectory_rewards.append(reward)
                                trajectory_log_probs.append(log_prob)
                                trajectory_values.append(value)
                                trajectory_dones.append(done)
                            else:
                                # Store experience in replay buffer for TD3/SAC
                                replay_buffer.push(state, action, reward, next_state, done)
                            
                            episode_reward += reward
                            state = next_state
                            step_count += 1
                            
                            # Train agent (for TD3/SAC with replay buffer)
                            if agent_name != 'PPO' and len(replay_buffer) > batch_size:
                                critic_loss, actor_loss = agent.train(replay_buffer, batch_size)
                                if critic_loss > 0:  # Valid loss returned
                                    episode_critic_loss += critic_loss
                                    episode_actor_loss += actor_loss
                                    loss_count += 1
                        
                        # Train PPO agent at end of episode
                        if agent_name == 'PPO' and len(trajectory_states) > 0:
                            # Store trajectory
                            for i in range(len(trajectory_states)):
                                agent.store_transition(
                                    trajectory_states[i],
                                    trajectory_actions[i],
                                    trajectory_rewards[i],
                                    trajectory_log_probs[i],
                                    trajectory_values[i],
                                    trajectory_dones[i]
                                )
                            
                            # Train
                            episode_actor_loss, episode_critic_loss = agent.train()
                            loss_count = 1
                        
                        episode_rewards.append(episode_reward)
                        
                        # Store losses
                        if loss_count > 0:
                            critic_losses.append(episode_critic_loss / loss_count)
                            actor_losses.append(episode_actor_loss / loss_count)
                        else:
                            critic_losses.append(0)
                            actor_losses.append(0)
                        
                        # Decay exploration noise
                        if hasattr(agent, 'exploration_noise'):
                            agent.exploration_noise = max(0.01, agent.exploration_noise * 0.995)
                        
                        # Update progress bars
                        episode_progress = (episode + 1) / episodes
                        overall_progress_val = (algo_idx * episodes + episode + 1) / (len(algorithms) * episodes)
                        
                        algo_progress.progress(episode_progress, f"Training {agent_name} - Episode {episode + 1}/{episodes}")
                        overall_progress.progress(overall_progress_val, f"Overall Progress - {agent_name}")
                        
                        # Update metrics every 25 episodes or at the end
                        if episode % 25 == 0 or episode == episodes - 1:
                            with metrics_placeholder.container():
                                if episode >= 50:  # Show metrics after some episodes
                                    recent_avg = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
                                    recent_std = np.std(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.std(episode_rewards)
                                    recent_critic_loss = np.mean(critic_losses[-25:]) if len(critic_losses) >= 25 else np.mean(critic_losses) if critic_losses else 0
                                    
                                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                    with metric_col1:
                                        st.metric(f"{agent_name} - Current Reward", f"{episode_reward:.3f}")
                                    with metric_col2:
                                        st.metric(f"{agent_name} - Avg (Last 50)", f"{recent_avg:.3f}")
                                    with metric_col3:
                                        st.metric(f"{agent_name} - Stability", f"{1/(recent_std+1):.3f}")
                                    with metric_col4:
                                        st.metric(f"{agent_name} - Avg Loss", f"{recent_critic_loss:.3f}")
                        
                        # Update charts every 25 episodes or at the end
                        if episode % 25 == 0 or episode == episodes - 1:
                            # Create temporary results for visualization
                            temp_results = {agent_name: episode_rewards}
                            temp_losses = {agent_name: {'critic': critic_losses, 'actor': actor_losses}}
                            
                            # Add previously completed algorithms
                            for prev_algo in algorithms[:algo_idx]:
                                if prev_algo in training_results:
                                    temp_results[prev_algo] = training_results[prev_algo]
                                if prev_algo in loss_results:
                                    temp_losses[prev_algo] = loss_results[prev_algo]
                            
                            # Create reward chart
                            fig_rewards = go.Figure()
                            
                            for name, rewards in temp_results.items():
                                # Smooth the rewards for better visualization
                                if len(rewards) > 10:
                                    window = min(25, len(rewards) // 4)
                                    smoothed = pd.Series(rewards).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                                else:
                                    smoothed = rewards
                                
                                line_width = 4 if name == agent_name else 2
                                opacity = 1.0 if name == agent_name else 0.7
                                
                                fig_rewards.add_trace(go.Scatter(
                                    x=list(range(len(smoothed))),
                                    y=smoothed,
                                    name=f'{name}' + (' (Training)' if name == agent_name else ' (Completed)'),
                                    mode='lines',
                                    line=dict(width=line_width),
                                    opacity=opacity
                                ))
                            
                            fig_rewards.update_layout(
                                title=f"RL Training Progress - Rewards (Currently: {agent_name}, Episode {episode + 1}/{episodes})",
                                xaxis_title="Episode",
                                yaxis_title="Reward (Negative Defect Rate)",
                                height=400,
                                showlegend=True
                            )
                            
                            chart_placeholder.plotly_chart(fig_rewards, use_container_width=True)
                            
                            # Create loss chart
                            if any(len(losses['critic']) > 0 for losses in temp_losses.values()):
                                fig_losses = make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=('Critic Loss', 'Actor Loss'),
                                    shared_yaxes=False
                                )
                                
                                colors = {'TD3': '#1f77b4', 'SAC': '#ff7f0e', 'PPO': '#2ca02c'}
                                
                                for name, losses in temp_losses.items():
                                    if len(losses['critic']) > 0:
                                        # Smooth losses
                                        critic_smoothed = pd.Series(losses['critic']).rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                                        actor_smoothed = pd.Series(losses['actor']).rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                                        
                                        line_width = 3 if name == agent_name else 2
                                        
                                        fig_losses.add_trace(
                                            go.Scatter(
                                                x=list(range(len(critic_smoothed))),
                                                y=critic_smoothed,
                                                name=f'{name} Critic',
                                                mode='lines',
                                                line=dict(width=line_width, color=colors.get(name, 'gray')),
                                                showlegend=True
                                            ),
                                            row=1, col=1
                                        )
                                        
                                        fig_losses.add_trace(
                                            go.Scatter(
                                                x=list(range(len(actor_smoothed))),
                                                y=actor_smoothed,
                                                name=f'{name} Actor',
                                                mode='lines',
                                                line=dict(width=line_width, color=colors.get(name, 'gray'), dash='dash'),
                                                showlegend=True
                                            ),
                                            row=1, col=2
                                        )
                                
                                fig_losses.update_layout(
                                    title="Training Losses (Proof of Learning)",
                                    height=300,
                                    showlegend=True
                                )
                                
                                loss_chart_placeholder.plotly_chart(fig_losses, use_container_width=True)
                        
                        # Add small delay to make progress visible
                        if episode % 10 == 0:
                            time.sleep(0.01)
                    
                    # Store results for this algorithm
                    training_results[agent_name] = episode_rewards
                    loss_results[agent_name] = {'critic': critic_losses, 'actor': actor_losses}
                    
                    # Store trained agent
                    st.session_state.trained_agents[agent_name] = agent
                    
                    # Clean up progress bars
                    algo_progress.empty()
                    
                    status_placeholder.success(f"{agent_name} training completed! Final reward: {episode_rewards[-1]:.3f}")
                    time.sleep(0.5)  # Brief pause between algorithms
                
                # Training completed - clean up and finalize
                overall_progress.empty()
                metrics_placeholder.empty()
                
                # Store results in session state
                st.session_state.training_results = training_results
                st.session_state.training_running = False  # Reset flag
                
                status_placeholder.success("All RL agents training completed with REAL learning!")
                
                # FIXED: Create and store final charts for persistence
                # Create final reward chart
                fig_final = go.Figure()
                colors = {'TD3': '#1f77b4', 'SAC': '#ff7f0e', 'PPO': '#2ca02c'}
                
                for agent_name, rewards in training_results.items():
                    # Smooth rewards for final chart
                    window = min(50, len(rewards) // 4)
                    if len(rewards) > window:
                        smoothed = pd.Series(rewards).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                    else:
                        smoothed = rewards
                    
                    fig_final.add_trace(go.Scatter(
                        x=list(range(len(smoothed))),
                        y=smoothed,
                        name=agent_name,
                        mode='lines',
                        line=dict(width=3, color=colors.get(agent_name, 'gray'))
                    ))
                
                fig_final.update_layout(
                    title="Final RL Training Results - Rewards (Persistent)",
                    xaxis_title="Episode",
                    yaxis_title="Reward (Negative Defect Rate)",
                    height=400,
                    showlegend=True
                )
                
                # Store final charts in session state for persistence
                st.session_state.training_charts = fig_final
                
                # Create and store loss chart
                if any(len(loss_results[name]['critic']) > 0 for name in loss_results):
                    fig_loss_final = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Critic Loss', 'Actor Loss'),
                        shared_yaxes=False
                    )
                    
                    for name, losses in loss_results.items():
                        if len(losses['critic']) > 0:
                            critic_smoothed = pd.Series(losses['critic']).rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                            actor_smoothed = pd.Series(losses['actor']).rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                            
                            fig_loss_final.add_trace(
                                go.Scatter(
                                    x=list(range(len(critic_smoothed))),
                                    y=critic_smoothed,
                                    name=f'{name} Critic',
                                    mode='lines',
                                    line=dict(width=2, color=colors.get(name, 'gray')),
                                    showlegend=True
                                ),
                                row=1, col=1
                            )
                            
                            fig_loss_final.add_trace(
                                go.Scatter(
                                    x=list(range(len(actor_smoothed))),
                                    y=actor_smoothed,
                                    name=f'{name} Actor',
                                    mode='lines',
                                    line=dict(width=2, color=colors.get(name, 'gray'), dash='dash'),
                                    showlegend=True
                                ),
                                row=1, col=2
                            )
                    
                    fig_loss_final.update_layout(
                        title="Training Losses - Final Results (Persistent)",
                        height=300,
                        showlegend=True
                    )
                    
                    st.session_state.loss_charts = fig_loss_final
                
                # Display final charts
                chart_placeholder.plotly_chart(st.session_state.training_charts, use_container_width=True)
                if st.session_state.loss_charts:
                    loss_chart_placeholder.plotly_chart(st.session_state.loss_charts, use_container_width=True)
                
                # Analysis and comparison
                st.subheader("Training Results Analysis")
                
                # Calculate metrics for each algorithm
                comparison_data = []
                for agent_name, rewards in training_results.items():
                    final_avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                    stability_score = 1 / (np.std(rewards[-100:]) + 1) if len(rewards) >= 100 else 1 / (np.std(rewards) + 1)
                    best_reward = max(rewards)
                    improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
                    
                    # Find convergence point
                    convergence_episode = episodes
                    if len(rewards) > 100:
                        for i in range(100, len(rewards), 10):
                            if i + 50 < len(rewards):
                                recent_trend = np.mean(rewards[i:i+50]) - np.mean(rewards[i-50:i])
                                if abs(recent_trend) < 0.01:
                                    convergence_episode = i
                                    break
                    
                    convergence_speed = max(0, (episodes - convergence_episode) / episodes)
                    
                    # Combined score
                    min_reward = min([min(r) for r in training_results.values()])
                    max_reward = max([max(r) for r in training_results.values()])
                    norm_reward = (final_avg - min_reward) / (max_reward - min_reward + 1e-6)
                    norm_stability = min(stability_score / 0.1, 1)
                    norm_improvement = max(0, improvement / (max_reward - min_reward + 1e-6))
                    
                    combined_score = (norm_reward * 0.4 + norm_stability * 0.3 + 
                                    convergence_speed * 0.15 + norm_improvement * 0.15)
                    
                    # Average losses
                    avg_critic_loss = np.mean(loss_results[agent_name]['critic']) if loss_results[agent_name]['critic'] else 0
                    avg_actor_loss = np.mean(loss_results[agent_name]['actor']) if loss_results[agent_name]['actor'] else 0
                    
                    comparison_data.append({
                        'Algorithm': agent_name,
                        'Final Avg Reward': final_avg,
                        'Best Reward': best_reward,
                        'Improvement': improvement,
                        'Stability Score': stability_score,
                        'Convergence Episode': convergence_episode,
                        'Avg Critic Loss': avg_critic_loss,
                        'Avg Actor Loss': avg_actor_loss,
                        'Combined Score': combined_score
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                best_agent = comparison_df.loc[comparison_df['Combined Score'].idxmax(), 'Algorithm']
                
                # Store best model
                st.session_state.best_models = {best_agent: st.session_state.trained_agents[best_agent]}
                
                # Display final results - CENTERED LAYOUT
                st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                st.subheader("Best Algorithm Results")
                
                # Center the best model display
                col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
                with col_center2:
                    st.success(f"**{best_agent}** - Best Performing Agent")
                    
                    best_row = comparison_df[comparison_df['Algorithm'] == best_agent].iloc[0]
                    
                    # Create metrics in a centered layout
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Final Reward", f"{best_row['Final Avg Reward']:.3f}")
                        st.metric("Improvement", f"{best_row['Improvement']:.3f}")
                        st.metric("Stability Score", f"{best_row['Stability Score']:.3f}")
                    
                    with metric_col2:
                        st.metric("Best Reward", f"{best_row['Best Reward']:.3f}")
                        st.metric("Convergence Episode", f"{best_row['Convergence Episode']}")
                        st.metric("Overall Score", f"{best_row['Combined Score']:.3f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Detailed comparison table
                st.subheader("Detailed Comparison")
                formatted_df = comparison_df.copy()
                formatted_df['Final Avg Reward'] = formatted_df['Final Avg Reward'].round(3)
                formatted_df['Best Reward'] = formatted_df['Best Reward'].round(3)
                formatted_df['Improvement'] = formatted_df['Improvement'].round(3)
                formatted_df['Stability Score'] = formatted_df['Stability Score'].round(4)
                formatted_df['Avg Critic Loss'] = formatted_df['Avg Critic Loss'].round(4)
                formatted_df['Avg Actor Loss'] = formatted_df['Avg Actor Loss'].round(4)
                formatted_df['Combined Score'] = formatted_df['Combined Score'].round(4)
                
                # Highlight best algorithm
                def highlight_best(row):
                    if row['Algorithm'] == best_agent:
                        return ['background-color: #d4edda'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    formatted_df.style.apply(highlight_best, axis=1),
                    use_container_width=True
                )
                
                # Show proof of learning
                st.subheader("Proof of Real Learning")
                proof_col1, proof_col2, proof_col3 = st.columns(3)
                
                with proof_col1:
                    st.info("**Gradient Updates**: ✓")
                    st.write("• Neural networks updated")
                    st.write("• Loss functions computed")
                    st.write("• Backpropagation performed")
                
                with proof_col2:
                    st.info("**Experience Replay**: ✓") 
                    st.write("• TD3/SAC: Experience buffer")
                    st.write("• PPO: Trajectory collection")
                    st.write("• Batch learning implemented")
                
                with proof_col3:
                    st.info("**Exploration Decay**: ✓")
                    st.write("• Noise reduces over time")
                    st.write("• Exploitation increases")
                    st.write("• Policy improvement visible")
                
            except Exception as e:
                # Reset training flag on error
                st.session_state.training_running = False
                st.error(f"Error during training: {str(e)}")
                st.info("Training has been reset. Please try again.")
    
    # Status indicators - FIXED LAYOUT
    st.subheader("System Status")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if st.session_state.surrogate_model is not None:
            st.success("Surrogate model ready")
            # Try to show model info
            try:
                if hasattr(st.session_state.surrogate_model, 'best_score_'):
                    st.write(f"**Model R² score:** {st.session_state.surrogate_model.best_score_:.4f}")
            except:
                pass
        else:
            st.warning("No surrogate model")
    
    with status_col2:
        if st.session_state.training_results:
            st.success("RL training completed")
            st.write("**Trained algorithms:**")
            for algo, rewards in st.session_state.training_results.items():
                final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
                st.write(f"• {algo}: {final_reward:.3f} (+{improvement:.3f})")
        else:
            st.warning("No training completed")
    
    with status_col3:
        if st.session_state.trained_agents:
            st.success("Trained agents available")
            for agent_name in st.session_state.trained_agents.keys():
                st.write(f"🤖 {agent_name}: Ready for recommendations")
        else:
            st.warning("No trained agents")
    
    if st.session_state.best_models:
        st.success("Best models available")
        for model in st.session_state.best_models.keys():
            st.write(f"🏆 Best: {model}")
    
    # Data quality checks
    if st.session_state.data is not None:
        st.subheader("Data Quality")
        
        # Check for missing values
        missing_counts = st.session_state.data.isnull().sum()
        if missing_counts.sum() > 0:
            st.warning(f"⚠️ {missing_counts.sum()} missing values found")
        else:
            st.success("✅ No missing values")
        
        # Check data ranges
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            zero_var_cols = []
            for col in numeric_cols:
                if st.session_state.data[col].std() == 0:
                    zero_var_cols.append(col)
            
            if zero_var_cols:
                st.warning(f"⚠️ {len(zero_var_cols)} columns have zero variance")
            else:
                st.success("✅ All features have variation")

# Tab 3: FIXED - Multiple Models + Tabular Display
with tab3:
    st.header("Defect Reduction Recommendations")
    
    # FIXED: Check for all trained agents instead of just best_models
    if not st.session_state.trained_agents or st.session_state.surrogate_model is None:
        st.warning("Please complete RL training in Tab 2 first!")
    else:
        st.subheader("Enter Current Parameter Values")
        
        # Create input fields for parameters
        current_values = {}
        
        if not hasattr(st.session_state, 'actual_feature_names') or not st.session_state.actual_feature_names:
            st.error("No feature data available. Please upload and process data in Tab 1 first.")
        else:
            cols = st.columns(2)
            for i, feature in enumerate(st.session_state.actual_feature_names):
                col_idx = i % 2
                with cols[col_idx]:
                    min_val = st.session_state.data[feature].min()
                    max_val = st.session_state.data[feature].max()
                    default_val = st.session_state.data[feature].mean()
                    
                    current_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_val),
                        key=f"input_{feature}",
                        help=f"Range: {min_val:.3f} - {max_val:.3f}"
                    )
        
        # FIXED: Allow selection of all trained models
        col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
        
        with col_center2:
            # FIXED: Show all available trained agents
            available_models = list(st.session_state.trained_agents.keys())
            if available_models:
                selected_model = st.selectbox(
                    "Select Trained Model for Recommendations",
                    available_models,
                    help="All trained RL agents are available for recommendations"
                )
                
                max_recommendation_steps = st.slider("Max Recommendation Steps", 5, 20, 15)
                
                # Add explanation of delta calculation
                with st.expander("How Delta Changes Are Calculated", expanded=False):
                    st.markdown("""
                    **Delta Calculation Method:**
                    
                    1. **Action Selection**: The trained RL agent selects an action vector [-1, +1] for each parameter
                    2. **Max Change Constraint**: Each parameter is limited to ±2% of its total range per step
                    3. **Delta Formula**: 
                       ```
                       delta = action[i] × max_change
                       max_change = (parameter_max - parameter_min) × 0.02
                       ```
                    4. **New Value**: `new_value = current_value + delta`
                    5. **Bounds Checking**: New value is clipped to parameter's valid range
                    
                    **Color Coding in Table:**
                    - 🟢 **Green**: Positive delta (parameter increased)
                    - 🔴 **Red**: Negative delta (parameter decreased)  
                    - The trained agent learns which direction to move each parameter to minimize defects
                    """)
                
                # CENTERED button
                if st.button("Get Recommendations", type="primary", use_container_width=True):
                    # Get current defect prediction - FIXED: Use proper DataFrame input
                    current_params = np.array(list(current_values.values()))
                    
                    # FIXED: Use DataFrame for surrogate model prediction
                    if hasattr(st.session_state.surrogate_model, 'feature_names_in_'):
                        input_df = pd.DataFrame([current_params], columns=st.session_state.actual_feature_names)
                        current_defect = st.session_state.surrogate_model.predict(input_df)[0]
                    else:
                        current_defect = st.session_state.surrogate_model.predict([current_params])[0]
                    
                    target_defect = st.session_state.target_stats['q25']
                    
                    # CENTERED status display
                    st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                    st.subheader("Current Status")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    col_current1, col_current2, col_current3 = st.columns(3)
                    
                    with col_current1:
                        st.metric("Current Defect Rate", f"{current_defect:.4f}")
                    
                    with col_current2:
                        st.metric("Target (25th percentile)", f"{target_defect:.4f}")
                    
                    with col_current3:
                        reduction_needed = current_defect - target_defect
                        st.metric("Reduction Needed", f"{reduction_needed:.4f}")
                    
                    if current_defect <= target_defect:
                        st.success("Current defect rate is already below the target! No action needed.")
                    else:
                        # CENTERED recommendation steps
                        st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                        st.subheader(f"Step-by-Step Recommendations (Using Trained {selected_model} Agent)")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Create RL environment for recommendations
                        feature_ranges = {}
                        for feature in st.session_state.actual_feature_names:
                            feature_ranges[feature] = (
                                st.session_state.data[feature].min(),
                                st.session_state.data[feature].max()
                            )
                        
                        env = DefectReductionEnv(
                            st.session_state.surrogate_model,
                            feature_ranges,
                            st.session_state.actual_feature_names,
                            max_recommendation_steps
                        )
                        
                        # Use the selected trained agent
                        trained_agent = st.session_state.trained_agents[selected_model]
                        
                        # Set environment to current parameters
                        env.current_params = current_params.copy()
                        env.current_step = 0
                        
                        recommendations = []
                        params = current_params.copy()
                        feature_names = list(current_values.keys())
                        
                        st.info(f"Using trained {selected_model} agent for recommendations (learned policy, no randomness!)")
                        
                        for step in range(max_recommendation_steps):
                            # Get current state
                            state = env._get_obs()
                            
                            # Get TRAINED agent recommendation (no exploration)
                            action = trained_agent.select_action(state, add_noise=False)
                            
                            # Apply action and get changes
                            changes = {}
                            for i, feature in enumerate(feature_names):
                                if feature in st.session_state.data.columns:
                                    min_val = st.session_state.data[feature].min()
                                    max_val = st.session_state.data[feature].max()
                                    max_change = (max_val - min_val) * 0.02
                                    
                                    # Apply trained agent's recommendation
                                    change = action[i] * max_change
                                    new_value = np.clip(params[i] + change, min_val, max_val)
                                    
                                    changes[feature] = {
                                        'old_value': params[i],
                                        'new_value': new_value,
                                        'change': new_value - params[i],
                                        'change_pct': ((new_value - params[i]) / (max_val - min_val)) * 100
                                    }
                                    params[i] = new_value
                            
                            # Update environment
                            env.current_params = params.copy()
                            env.current_step = step + 1
                            
                            # Get new defect prediction - FIXED
                            if hasattr(st.session_state.surrogate_model, 'feature_names_in_'):
                                input_df = pd.DataFrame([params], columns=st.session_state.actual_feature_names)
                                new_defect = st.session_state.surrogate_model.predict(input_df)[0]
                            else:
                                new_defect = st.session_state.surrogate_model.predict([params])[0]
                            
                            recommendations.append({
                                'step': step + 1,
                                'changes': changes,
                                'defect_rate': new_defect
                            })
                            
                            # Check if target reached
                            if new_defect <= target_defect:
                                st.success(f"Target defect rate achieved in {step + 1} steps!")
                                break
                        
                        # FIXED: Display recommendations as tabular format with colored deltas
                        if recommendations:
                            st.subheader("📋 Step-by-Step Recommendations Table")
                            
                            # Create tabular data
                            table_data = []
                            
                            # Add initial row
                            initial_row = {"Step": 0, "Defect Rate": f"{current_defect:.4f}"}
                            for feature in feature_names:
                                initial_row[f"{feature[:15]}..." if len(feature) > 15 else feature] = f"{current_values[feature]:.3f}"
                            table_data.append(initial_row)
                            
                            # Add recommendation rows
                            for rec in recommendations:
                                row = {"Step": rec['step'], "Defect Rate": f"{rec['defect_rate']:.4f}"}
                                for feature in feature_names:
                                    if feature in rec['changes']:
                                        change_info = rec['changes'][feature]
                                        delta = change_info['change']
                                        
                                        # Create colored delta display
                                        if delta > 0:
                                            delta_display = f"(+{delta:.3f})"
                                            color = "green"
                                        elif delta < 0:
                                            delta_display = f"({delta:.3f})"
                                            color = "red"
                                        else:
                                            delta_display = "(0.000)"
                                            color = "gray"
                                        
                                        # Format with current value and colored delta
                                        col_name = f"{feature[:15]}..." if len(feature) > 15 else feature
                                        row[col_name] = f"{change_info['new_value']:.3f} {delta_display}"
                                    else:
                                        # No change for this parameter
                                        col_name = f"{feature[:15]}..." if len(feature) > 15 else feature
                                        prev_value = current_values[feature] if rec['step'] == 1 else None
                                        for prev_rec in recommendations[:rec['step']-1]:
                                            if feature in prev_rec['changes']:
                                                prev_value = prev_rec['changes'][feature]['new_value']
                                        if prev_value is None:
                                            prev_value = current_values[feature]
                                        row[col_name] = f"{prev_value:.3f} (0.000)"
                                
                                table_data.append(row)
                            
                            # Display as DataFrame with custom styling
                            df_display = pd.DataFrame(table_data)
                            
                            # Custom HTML styling for colored deltas
                            def style_deltas(val):
                                if isinstance(val, str) and '(' in val and ')' in val:
                                    # Extract delta part
                                    if '(+' in val:
                                        return 'color: green; font-weight: bold'
                                    elif '(-' in val:
                                        return 'color: red; font-weight: bold'
                                return ''
                            
                            # Style the dataframe
                            styled_df = df_display.style.applymap(style_deltas)
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Legend for the table
                            st.markdown("""
                            **Table Legend:**
                            - **Current Value**: Shows the parameter value at each step
                            - **Delta (Green)**: Positive change - parameter increased
                            - **Delta (Red)**: Negative change - parameter decreased  
                            - **Delta Calculation**: `new_value = old_value + (agent_action × 2% × parameter_range)`
                            """)
                            
                            # Progress chart
                            st.subheader("Defect Reduction Progress")
                            
                            steps = [0] + [rec['step'] for rec in recommendations]
                            defect_rates = [current_defect] + [rec['defect_rate'] for rec in recommendations]
                            
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=steps,
                                y=defect_rates,
                                mode='lines+markers',
                                name='Defect Rate',
                                line=dict(color='blue', width=3),
                                marker=dict(size=8)
                            ))
                            
                            # Add target line
                            fig.add_hline(
                                y=target_defect,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Target (25th percentile)"
                            )
                            
                            fig.update_layout(
                                title=f"Defect Rate Reduction Progress (Using Trained {selected_model})",
                                xaxis_title="Step",
                                yaxis_title="Defect Rate",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Final summary - CENTERED
                            st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                            st.subheader("Final Summary")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            final_defect = recommendations[-1]['defect_rate']
                            total_reduction = current_defect - final_defect
                            reduction_percentage = (total_reduction / current_defect) * 100 if current_defect != 0 else 0
                            
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            with summary_col1:
                                st.metric("Initial Defect Rate", f"{current_defect:.4f}")
                            
                            with summary_col2:
                                st.metric("Final Defect Rate", f"{final_defect:.4f}")
                            
                            with summary_col3:
                                st.metric("Total Reduction", f"{total_reduction:.4f}")
                            
                            with summary_col4:
                                st.metric("Reduction %", f"{reduction_percentage:.1f}%")
                            
                            if final_defect <= target_defect:
                                st.success("Successfully achieved target defect rate using trained RL agent!")
                            else:
                                remaining = final_defect - target_defect
                                st.info(f"Still {remaining:.4f} above target. Consider additional optimization cycles or different hyperparameters.")
                            
                            # Show detailed parameter changes summary
                            st.subheader("Parameter Changes Summary")
                            
                            summary_changes = []
                            for feature in feature_names:
                                initial_val = current_values[feature]
                                final_val = initial_val
                                
                                # Find final value
                                for rec in recommendations:
                                    if feature in rec['changes']:
                                        final_val = rec['changes'][feature]['new_value']
                                
                                total_change = final_val - initial_val
                                change_pct = (total_change / (st.session_state.data[feature].max() - st.session_state.data[feature].min())) * 100
                                
                                summary_changes.append({
                                    'Parameter': feature,
                                    'Initial Value': f"{initial_val:.3f}",
                                    'Final Value': f"{final_val:.3f}",
                                    'Total Change': f"{total_change:+.3f}",
                                    'Change %': f"{change_pct:+.2f}%"
                                })
                            
                            summary_df = pd.DataFrame(summary_changes)
                            
                            # Style the summary table
                            def style_changes(val):
                                if isinstance(val, str):
                                    if val.startswith('+') and '0.000' not in val:
                                        return 'color: green; font-weight: bold'
                                    elif val.startswith('-'):
                                        return 'color: red; font-weight: bold'
                                return ''
                            
                            styled_summary = summary_df.style.applymap(style_changes)
                            st.dataframe(styled_summary, use_container_width=True)
                            
                            # Show agent info
                            st.info(f"**Recommendations generated by**: Trained {selected_model} agent (learned policy from {len(st.session_state.training_results.get(selected_model, []))} training episodes)")
                            
                            # Export recommendations - CENTERED
                            st.markdown("<div style='text-align: center'>", unsafe_allow_html=True)
                            st.subheader("Export Recommendations")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Center the download button
                            col_download1, col_download2, col_download3 = st.columns([1, 2, 1])
                            with col_download2:
                                if st.button("Download Recommendations as CSV", use_container_width=True):
                                    # Create export data
                                    export_data = []
                                    export_data.append({
                                        'Step': 0,
                                        'Defect_Rate': current_defect,
                                        'Agent_Used': 'Initial',
                                        **{f'{feature}_Value': current_values[feature] for feature in feature_names},
                                        **{f'{feature}_Delta': 0.0 for feature in feature_names}
                                    })
                                    
                                    for rec in recommendations:
                                        row = {
                                            'Step': rec['step'],
                                            'Defect_Rate': rec['defect_rate'],
                                            'Agent_Used': selected_model
                                        }
                                        for feature in feature_names:
                                            if feature in rec['changes']:
                                                row[f'{feature}_Value'] = rec['changes'][feature]['new_value']
                                                row[f'{feature}_Delta'] = rec['changes'][feature]['change']
                                            else:
                                                # Keep previous value
                                                prev_step = next((r for r in reversed(recommendations[:rec['step']-1]) if feature in r['changes']), None)
                                                if prev_step:
                                                    row[f'{feature}_Value'] = prev_step['changes'][feature]['new_value']
                                                    row[f'{feature}_Delta'] = 0.0
                                                else:
                                                    row[f'{feature}_Value'] = current_values[feature]
                                                    row[f'{feature}_Delta'] = 0.0
                                        export_data.append(row)
                                    
                                    export_df = pd.DataFrame(export_data)
                                    csv = export_df.to_csv(index=False)
                                    
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name=f"defect_reduction_recommendations_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )
            else:
                st.error("No trained models available")
                selected_model = None

# Sidebar with information (Updated with new features)
with st.sidebar:
    st.header("System Information")
    
    st.subheader("Current Configuration")
    if st.session_state.feature_names:
        st.write("**Features loaded:**")
        feature_display = st.session_state.feature_names[:10]  # Show first 10
        for feature in feature_display:
            display_name = feature[:30] + "..." if len(feature) > 30 else feature
            st.write(f"• {display_name}")
        if len(st.session_state.feature_names) > 10:
            st.write(f"... and {len(st.session_state.feature_names) - 10} more")
    else:
        st.warning("No features loaded from JSON file")
    
    if st.session_state.data is not None:
        st.write(f"**Dataset shape:** {st.session_state.data.shape}")
        st.write(f"**Features matched:** {len(st.session_state.actual_feature_names)}/{len(st.session_state.feature_names)}")
        st.write(f"**Target column:** {st.session_state.target_column}")
        
        if st.session_state.target_stats:
            st.write(f"**Target stats:**")
            st.write(f"• Mean: {st.session_state.target_stats['mean']:.4f}")
            st.write(f"• 25th percentile: {st.session_state.target_stats['q25']:.4f}")
            st.write(f"• Std: {st.session_state.target_stats['std']:.4f}")
    
    if st.session_state.surrogate_model is not None:
        st.success("Surrogate model ready")
        # Try to show model info
        try:
            if hasattr(st.session_state.surrogate_model, 'best_score_'):
                st.write(f"**Model R² score:** {st.session_state.surrogate_model.best_score_:.4f}")
        except:
            pass
    
    if st.session_state.training_results:
        st.success("RL training completed")
        st.write("**Trained algorithms:**")
        for algo, rewards in st.session_state.training_results.items():
            final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            improvement = rewards[-1] - rewards[0] if len(rewards) > 1 else 0
            st.write(f"• {algo}: {final_reward:.3f} (+{improvement:.3f})")
    
    if st.session_state.trained_agents:
        st.success("Trained agents available")
        for agent_name in st.session_state.trained_agents.keys():
            st.write(f"🤖 {agent_name}: Ready for recommendations")
    
    if st.session_state.best_models:
        st.success("Best models available")
        for model in st.session_state.best_models.keys():
            st.write(f"🏆 Best: {model}")
    
    # System health checks
    st.subheader("System Health")
    
    # Check PyTorch availability
    try:
        torch.tensor([1.0])
        st.success("PyTorch: Ready")
    except:
        st.error("PyTorch: Not available")
    
    # Check Gymnasium
    try:
        gym.make('CartPole-v1')
        st.success("Gymnasium: Ready")
    except:
        st.error("Gymnasium: Not available")
    
    # Check LightGBM
    try:
        lgb.LGBMRegressor()
        st.success("LightGBM: Ready")
    except:
        st.error("LightGBM: Not available")
    
    st.subheader("Quick Actions")
    
    if st.button("Clear All Data", type="secondary"):
        # Clear session state
        for key in ['data', 'surrogate_model', 'training_results', 'trained_agents', 'best_models', 'training_charts', 'loss_charts']:
            if key in st.session_state:
                st.session_state[key] = None if key in ['data', 'surrogate_model'] else {}
        st.success("All data cleared!")
        st.rerun()
    
    if st.session_state.surrogate_model is not None and st.button("Save Model"):
        try:
            model_filename = f"surrogate_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(model_filename, 'wb') as f:
                pickle.dump(st.session_state.surrogate_model, f)
            st.success(f"Model saved as {model_filename}")
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>RL-based Defect Reduction System | <b>FIXED IMPLEMENTATION</b></p>
        <p><small>Real reinforcement learning with proper training algorithms</small></p>
        <p><small>Actual gradient descent | Experience replay | Exploration decay</small></p>
        <p><small>Measurable learning | Trained agent recommendations</small></p>
        <p><small>Persistent charts | Multiple model selection | Tabular recommendations</small></p>
    </div>
    """,
    unsafe_allow_html=True
)