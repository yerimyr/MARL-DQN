import numpy as np
from config_SimPy import *
from config_DQN import *
from log_DQN import *
from environment import *
from MARL_DQN import *


class GymWrapper:
    """
    Wrapper class to handle the interaction between MARL_DQN and Gym environment

    Args:
        env (gym.Env): Gym environment
        n_agents (int): Number of agents in the environment
        action_dim (int): Dimension of the action space
        state_dim (int): Dimension of the state space
        buffer_size (int): Size of the replay buffer
        batch_size (int): Batch size for training (unit: episodes)
        lr (float): Learning rate for the actor and critic networks
        gamma (float): Discount factor for future rewards
        hidden_dim (int): Hidden dimension for actor and critic networks
    """

    def __init__(self, env, n_agents, action_dim, state_dim, buffer_size, batch_size, lr, gamma):
        self.env = env
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size

        # Initialize MARL_DQN components with correct parameter order
        self.dqn = MultiAgentDQN(
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma
        )
        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # Initialize tensorboard logger
        self.logger = TensorboardLogger(n_agents)
        '''
        # Log hyperparameters
        self.logger.log_hyperparameters({
            'n_agents': n_agents,
            'action_dim': action_dim,
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'learning_rate': lr,
            'gamma': gamma
        })
        '''
    def train(self, episodes, eval_interval):
        """
        Train the MARL_DQN system using the Gym environment

        Args:
            episodes: Number of training episodes
            eval_interval: Interval for evaluation and printing results
        """
        best_reward = float('-inf')  # best_reward 변수 초기화. 
        action_history = {i: [] for i in range(self.n_agents)}  # agent별 action 저장
        q_value_history = {i: [] for i in range(self.n_agents)}  # agent별 Q-value 저장
         
        for episode in range(episodes):
            states = self.env.reset()
            episode_reward = 0
            done = False
            epsilon = max(0.1, 1.0 - episode/500)  # 탐험률 설정: 입실론은 초기에는 1.0으로 시작하여 0.1까지 선형적으로 감소함.

            q_values = []  # 이번 episode에서 agent들의 Q-value 저장
            while not done:
                # Select actions for each agent
                actions = []
                step_q_values = []  # 각 step에서 agent별 Q-value 저장
                for i in range(self.n_agents):
                    action, q_value = self.dqn.select_action(states[i], epsilon)  # action과 Q-value 반환
                    actions.append(action)
                    action_history[i].append((episode, action))
                    if q_value is not None:
                        step_q_values.append(q_value)

                # Execute actions in environment
                next_states, reward, done, info = self.env.step(actions)  # 선택된 행동들을 환경에 전달하여 얻는 정보들.

                # Store transition in buffer
                self.buffer.push(states, np.array(actions),
                                 reward, next_states, done)

                episode_reward += reward
                states = next_states
                
                if step_q_values:
                    q_values.append(np.mean(step_q_values))  # step마다 평균 Q-value 저장


                # Print simulation events
                if PRINT_SIM_EVENTS:
                    print(info)

            # If we have enough complete episodes, perform training
            if len(self.buffer) >= self.batch_size:
                self.dqn.update(self.batch_size)
            
            self.logger.log_agent_action_distribution(episode, actions)
            
            # Q-value TensorBoard에 기록 
            avg_q_value = np.mean(q_values) if q_values else 0
            self.logger.log_q_values(episode, [avg_q_value] * self.n_agents)

            self.logger.log_training_info(
                episode, episode_reward, -episode_reward/self.env.current_day, info['inventory_levels'], epsilon
            )

            # 각 에피소드가 끝날 때마다 Replay Buffer의 크기를 tensorboard에 기록.
            self.logger.log_replay_buffer_size(episode, len(self.buffer))
            
            # 기존 로그 기능 유지
            avg_cost = -episode_reward / self.env.current_day
            self.logger.log_training_info(episode, episode_reward, avg_cost, info['inventory_levels'], epsilon)

            # eval_interval마다 현재 학습 상태를 콘솔에 출력.
            if episode % eval_interval == 0:
                print(f"Replay Buffer Size: {len(self.buffer)}")
                
            # Log training information
            avg_cost = -episode_reward/self.env.current_day
            self.logger.log_training_info(
                episode=episode,
                episode_reward=episode_reward,
                avg_cost=avg_cost,
                inventory_levels=info['inventory_levels'],
                epsilon=epsilon
            )

            # Evaluation and saving best model
            if episode % eval_interval == 0:
                print(f"Episode {episode}")
                print(f"Episode Reward: {episode_reward}")
                print(f"Average Cost: {avg_cost}")
                print("Inventory Levels:", info['inventory_levels'])
                print("-" * 50)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.save_model(episode, episode_reward)
    
        return action_history  # Q-value 기록은 TensorBoard에서 확인 가능
    
    def evaluate(self, episodes):
        """
        Evaluate the trained MAAC system

        Args:
            episodes: Number of evaluation episodes
        """
        for episode in range(episodes):
            observations = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select actions without exploration
                actions = []
                for i in range(self.n_agents):
                    # observations에서 인덱스를 통해 접근
                    local_obs = observations[i]
                    action = self.dqn.select_action(local_obs, epsilon=0)
                    print(f"Agent {i}: Action selected {action}")

                observations, reward, done, info = self.env.step(actions)
                episode_reward += reward

                self.env.render()  # Visualize the environment state

            avg_daily_cost = -episode_reward/self.env.current_day

            # Log evaluation information
            self.logger.log_evaluation_info(
                episode=episode,
                total_reward=episode_reward,
                avg_daily_cost=avg_daily_cost,
                inventory_levels=info['inventory_levels']
            )

            print(f"Evaluation Episode {episode}")
            print(f"Total Reward: {episode_reward}")
            print(f"Average Daily Cost: {avg_daily_cost}")
            print("-" * 50)

    def save_model(self, episode, reward):
        """
        Save the model to the specified path

        Args: 
            episode (int): Current episode number
            reward (float): Current episode reward 
        """
        # Save best model
        model_path = os.path.join(
            MODEL_DIR, f"dqn_best_model_episode_{episode}.pt")
        torch.save({
            'episode': episode,
            'best_reward': reward,
            'q_network_state_dict': self.dqn.q_network.state_dict(),
            'target_q_network_state_dict': self.dqn.target_network.state_dict(),
            'optimizer_state_dict': self.dqn.optimizer.state_dict()
        }, model_path)
        #print(f"Saved best model with reward {reward} to {model_path}")

    def load_model(self, model_path):
        """
        Load a saved model

        Args:
            model_path (str): Path to the saved model
        """
        checkpoint = torch.load(model_path)

        # Load model states
        self.dqn.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.dqn.target_network.load_state_dict(checkpoint['target_q_network_state_dict'])

        # Load optimizer state
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded model from episode {checkpoint['episode']} with best reward {checkpoint['best_reward']}")

    def __del__(self):
        """Cleanup method to ensure tensorboard writer is closed"""
        if hasattr(self, 'logger'):
            self.logger.close()
