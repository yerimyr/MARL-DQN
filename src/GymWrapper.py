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

        self.buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )
        
        # Initialize MARL_DQN components with correct parameter order
        self.dqn = MultiAgentDQN(
            n_agents=n_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            buffer=self.buffer,
            lr=lr,
            gamma=gamma
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
        best_reward = float('-inf')  
        action_history = {i: [] for i in range(self.n_agents)}  # agent별 action 저장
         
        for episode in range(episodes):
            states = self.env.reset()
            episode_reward = 0
            done = False
            epsilon = max(0.1, 1.0 - episode/500)  # 입실론 변화: 입실론은 초기에는 1.0으로 시작하여 0.1까지 선형적으로 감소함.

            while not done:
                # Select actions for each agent
                actions = []
                for i in range(self.n_agents):
                    action = self.dqn.select_action(states[i], epsilon)  # 각 에이전트가 자신의 state만 참고하여 action을 선택함. (Centralized Execution)
                    actions.append(action)
                    action_history[i].append((episode, action))

                # Execute actions in environment
                next_states, reward, done, info = self.env.step(actions)  

                # Store transition in buffer
                self.buffer.push(states, np.array(actions),
                                 reward, next_states, done)

                episode_reward += reward
                states = next_states

                # Print simulation events
                if PRINT_SIM_EVENTS:
                    print(info)

            # If we have enough complete episodes, perform training
            if len(self.buffer) >= self.batch_size:
                self.dqn.update(self.batch_size)
            
            #self.logger.log_agent_action(episode, actions)
            
            self.logger.log_training_info(
                episode, episode_reward, -episode_reward/self.env.current_day, info['inventory_levels'], epsilon
            )

            # 각 에피소드가 끝날 때마다 Replay Buffer의 크기를 tensorboard에 기록.
            self.logger.log_replay_buffer_size(episode, len(self.buffer))
                
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
