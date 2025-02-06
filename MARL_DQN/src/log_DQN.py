from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from datetime import datetime
from config_DQN import *


# Log daily repots: Inventory level for each item; In-transition inventory for each material; Remaining demand (demand - product level)
STATE_ACTION_REPORT_REAL = [[]]  # Real State
COST_RATIO_HISTORY = []

# Record the cumulative value of each cost component
LOG_TOTAL_COST_COMP = {
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}


class TensorboardLogger:
    """
    Tensorboard logging utility for MAAC training

    Args:
        log_dir (str): Directory to save tensorboard logs
        n_agents (int): Number of agents in the environment
    """

    def __init__(self, n_agents, log_dir='runs'):
        # Create unique run name with timestamp
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_name = f'MARL_DQN_run_{current_time}'

        # Create log directory
        self.log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(self.log_dir)
        self.n_agents = n_agents

        print(f"Tensorboard logging to {self.log_dir}")
        print("To view training progress, run:")
        print(f"tensorboard --logdir={log_dir}")

    def log_training_info(self, episode, episode_reward, avg_cost, inventory_levels,
                          epsilon=None):  # 학습 중 에피소드 단위로 데이터(reward, avg_cost, inventory_levels, epsilon) 기록
        """
        Log training metrics to tensorboard

        Args:
            episode (int): Current episode number
            episode_reward (float): Total reward for the episode
            avg_cost (float): Average cost per day for the episode
            inventory_levels (dict): Dictionary of inventory levels for each agent
            critic_loss (float, optional): Loss value of the critic network
            actor_losses (list, optional): List of loss values for each actor network
            epsilon (float, optional): Current exploration rate
        """
        # Log episode metrics
        self.writer.add_scalar('Training/Episode_Reward',
                               episode_reward, episode)
        self.writer.add_scalar(
            'Training/Average_Daily_Cost', avg_cost, episode)

        # Log inventory levels for each agent
        for agent_id, level in inventory_levels.items():
            self.writer.add_scalar(
                f'Inventory/Agent_{agent_id}', level, episode)
        
        # Log exploration rate if provided
        if epsilon is not None:
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)
        
        
    
    def log_evaluation_info(self, episode, total_reward, avg_daily_cost, inventory_levels):  # 평가 중 데이터 기록록
        """
        Log evaluation metrics to tensorboard

        Args:
            episode (int): Current evaluation episode
            total_reward (float): Total reward for the evaluation episode
            avg_daily_cost (float): Average daily cost during evaluation
            inventory_levels (dict): Dictionary of inventory levels for each agent
        """
        self.writer.add_scalar(
            'Evaluation/Episode_Reward', total_reward, episode)
        self.writer.add_scalar(
            'Evaluation/Average_Daily_Cost', avg_daily_cost, episode)

        # Log evaluation inventory levels
        for agent_id, level in inventory_levels.items():
            self.writer.add_scalar(f'Evaluation/Inventory_Agent_{agent_id}',
                                   level, episode)
            
    
    def log_hyperparameters(self, config_dict):  # 하이퍼파라미터 설정 정보 기록
        """
        Log hyperparameters to tensorboard

        Args:
            config_dict (dict): Dictionary containing hyperparameters
        """
        # Log hyperparameters as text
        hyperparams_text = "\n".join(
            [f"{k}: {v}" for k, v in config_dict.items()])
        self.writer.add_text('Hyperparameters', hyperparams_text)

        # Also log as hparams for the experiment comparison interface
        self.writer.add_hparams(config_dict, {'hparam/dummy_metric': 0})
    
    # validation
    def log_replay_buffer_size(self, episode, buffer_size):
        """
        Replay Buffer 크기를 TensorBoard에 기록
        Args:
            episode (int): 현재 에피소드
            buffer_size (int): Replay Buffer 크기
        """
        self.writer.add_scalar('Training/Replay_Buffer_Size', buffer_size, episode)
        
    # validation
    def log_agent_action_distribution(self, episode, agent_actions):
        """
        각 에이전트별 선택한 Action 분포를 TensorBoard에 기록 (히스토그램 사용)
        Args:
            episode (int): 현재 에피소드
            agent_actions (list): 각 에이전트가 선택한 액션 리스트 (길이: n_agents)
        """
        for agent_id, action in enumerate(agent_actions):
            self.writer.add_histogram(f'Agent_Actions/Agent_{agent_id+1}', action, episode, bins=6)
 
            
    # validation
    def log_q_values(self, episode, avg_q_values):
        """
        선택한 action의 Q-value를 TensorBoard에 기록
        """
        for agent_id, q_value in enumerate(avg_q_values):
            if q_value is not None:  # Q-value가 존재하는 경우만 기록
                self.writer.add_scalar(f'Q_Values/Agent_{agent_id+1}', q_value, episode)

    def close(self):
        """Close the tensorboard writer"""
        self.writer.close()