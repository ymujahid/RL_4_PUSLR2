import pygame
import numpy as np
import random
import math
import time
import json
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime

class PULSR2GUI:
    def __init__(self, environment):
        """Initialize GUI for PULSR2 environment visualization"""
        # Store reference to environment
        self.env = environment
        
        # Initialize pygame
        pygame.init()
        
        # Define colors
        self.colors = {
            'background': (255, 255, 255),  # White
            'end_effector': (255, 0, 0),    # Red
            'target': (0, 255, 0),          # Green
            'text': (0, 0, 0),              # Black
            'grid': (200, 200, 200),        # Light gray
            'trajectory': (100, 100, 255),  # Light blue
            'completed': (50, 200, 50)      # Light green
        }
        
        # Set up display dimensions
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('PULSR2 Reinforcement Learning Environment')
        
        # Set up scaling factors to map environment coordinates to screen coordinates
        self.scale_factor = 5  # Pixels per unit in environment
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height // 2
        
        # Set up fonts
        self.font = pygame.font.SysFont('Arial', 18)
        
        # Initialize clock for controlling frame rate
        self.clock = pygame.time.Clock()
    
    def env_to_screen_coords(self, x, y):
        """Convert environment coordinates to screen coordinates"""
        screen_x = self.origin_x + x * self.scale_factor
        screen_y = self.origin_y - y * self.scale_factor  # Invert y-axis
        return int(screen_x), int(screen_y)
    
    def draw_trajectory(self):
        """Draw the curved trajectory"""
        # Draw lines connecting all trajectory points
        for i in range(len(self.env.trajectory_points) - 1):
            # Convert trajectory points to screen coordinates
            x1, y1 = self.env.trajectory_points[i]
            x2, y2 = self.env.trajectory_points[i+1]
            start_pos = self.env_to_screen_coords(x1, y1)
            end_pos = self.env_to_screen_coords(x2, y2)
            
            # Draw line segment with different color based on completion
            if i < self.env.current_target_idx - 1:
                # Completed segments in light green
                color = self.colors['completed']
                width = 3
            else:
                # Upcoming segments in light blue
                color = self.colors['trajectory']
                width = 2
            
            pygame.draw.line(self.screen, color, start_pos, end_pos, width)
            
        # Draw small circles at each waypoint
        for i, (x, y) in enumerate(self.env.trajectory_points):
            screen_pos = self.env_to_screen_coords(x, y)
            if i < self.env.current_target_idx:
                # Completed waypoints in green
                color = self.colors['completed']
            else:
                # Upcoming waypoints in blue
                color = self.colors['trajectory']
            
            pygame.draw.circle(self.screen, color, screen_pos, 3)
    
    def draw_end_effector(self):
        """Draw the end effector"""
        x = self.env.state_space['end_effector_x']
        y = self.env.state_space['end_effector_y']
        screen_x, screen_y = self.env_to_screen_coords(x, y)
        
        # Draw end effector as a circle
        pygame.draw.circle(self.screen, self.colors['end_effector'], 
                           (screen_x, screen_y), 10)
    
    def draw_target(self):
        """Draw the target position"""
        x = self.env.state_space['target_x']
        y = self.env.state_space['target_y']
        screen_x, screen_y = self.env_to_screen_coords(x, y)
        
        # Draw target as a circle with crosshairs
        pygame.draw.circle(self.screen, self.colors['target'], 
                           (screen_x, screen_y), 10, 2)
        pygame.draw.line(self.screen, self.colors['target'], 
                         (screen_x - 15, screen_y), (screen_x + 15, screen_y), 2)
        pygame.draw.line(self.screen, self.colors['target'], 
                         (screen_x, screen_y - 15), (screen_x, screen_y + 15), 2)
    
    def draw_info_panel(self, episode, step, reward, total_reward):
        """Draw information panel with current statistics"""
        texts = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Current Reward: {reward:.2f}",
            f"Total Reward: {total_reward:.2f}",
            f"End Effector: ({self.env.state_space['end_effector_x']:.1f}, {self.env.state_space['end_effector_y']:.1f})",
            f"Target: ({self.env.state_space['target_x']:.1f}, {self.env.state_space['target_y']:.1f})",
            f"Waypoint: {self.env.current_target_idx}/{len(self.env.trajectory_points)}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (10, 10 + i * 25))
    
    def render(self, episode=0, step=0, reward=0, total_reward=0):
        """Render the current state of the environment"""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw trajectory
        self.draw_trajectory()
        
        # Draw target and end effector
        self.draw_target()
        self.draw_end_effector()
        
        # Draw information panel
        self.draw_info_panel(episode, step, reward, total_reward)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(30)
    
    def check_events(self):
        """Check for pygame events (e.g., window close)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    
    def close(self):
        """Close the GUI"""
        pygame.quit()

class PULSR2Environment:
    def __init__(self):
        # State Space Definition
        self.state_space = {
            'end_effector_x': 0,     # X coordinate of end effector
            'end_effector_y': 0,     # Y coordinate of end effector
            'target_x': 30,          # Target X coordinate
            'target_y': 20           # Target Y coordinate
        }
        
        # Action Space Definition (4 possible movements)
        self.actions = {
            0: 'motion1',  # North direction (upper motor forward)
            1: 'motion2',  # South direction (upper motor reverse)
            2: 'motion3',  # East direction (lower motor forward)
            3: 'motion4'   # West direction (lower motor reverse)
        }
        
        # Movement step size
        self.step_size = 5
        
        # Workspace boundaries
        self.workspace_bounds = (-50, 50)
        
        # Trajectory parameters
        self.trajectory_points = self.generate_curved_trajectory()
        self.current_target_idx = 0
        self.trajectory_completed = False
        
    def generate_curved_trajectory(self):
        """Generate a curved trajectory using a sine wave pattern"""
        points = []
        # Create a curved path from left to right
        for x in range(-40, 41, 5):
            # Use sine function to create the curve
            y = 20 * math.sin(x * 0.1)
            points.append((x, y))
        return points
        
    def reset(self):
        """Reset environment to initial state"""
        # Initialize end effector position to start of trajectory
        self.state_space['end_effector_x'] = self.trajectory_points[0][0]
        self.state_space['end_effector_y'] = self.trajectory_points[0][1]
        
        # Set first target point
        self.current_target_idx = 1
        self.state_space['target_x'] = self.trajectory_points[self.current_target_idx][0]
        self.state_space['target_y'] = self.trajectory_points[self.current_target_idx][1]
        
        self.trajectory_completed = False
        
        return self.get_state()
        
    def step(self, action):
        """Execute action and return next state, reward, done"""
        # Execute motion based on action
        if action in self.actions:
            self.execute_motion(action)
            
        # Calculate reward
        reward = self._get_reward()
        
        # Check if current target is reached
        current_distance = math.sqrt(
            (self.state_space['end_effector_x'] - self.state_space['target_x'])**2 +
            (self.state_space['end_effector_y'] - self.state_space['target_y'])**2
        )
        
        # If target reached, move to next target point
        if current_distance < 5:
            self.current_target_idx += 1
            # Check if we've reached the end of trajectory
            if self.current_target_idx >= len(self.trajectory_points):
                self.trajectory_completed = True
            else:
                # Set next target
                self.state_space['target_x'] = self.trajectory_points[self.current_target_idx][0]
                self.state_space['target_y'] = self.trajectory_points[self.current_target_idx][1]
        
        # Check if episode is done
        done = self._is_terminal()
        
        return self.get_state(), reward, done
    
    def execute_motion(self, action):
        """Execute a motion based on the action"""
        if action == 0:  # North
            self.state_space['end_effector_y'] += self.step_size
        elif action == 1:  # South
            self.state_space['end_effector_y'] -= self.step_size
        elif action == 2:  # East
            self.state_space['end_effector_x'] += self.step_size
        elif action == 3:  # West
            self.state_space['end_effector_x'] -= self.step_size
            
        # Ensure we stay within workspace bounds
        self.state_space['end_effector_x'] = max(self.workspace_bounds[0], 
                                               min(self.workspace_bounds[1], 
                                                   self.state_space['end_effector_x']))
        self.state_space['end_effector_y'] = max(self.workspace_bounds[0], 
                                               min(self.workspace_bounds[1], 
                                                   self.state_space['end_effector_y']))
        
    def _get_reward(self):
        """Calculate reward based on distance to target and trajectory following"""
        current_distance = math.sqrt(
            (self.state_space['end_effector_x'] - self.state_space['target_x'])**2 +
            (self.state_space['end_effector_y'] - self.state_space['target_y'])**2
        )
        
        # Base reward based on distance to current target
        if current_distance < 5:  # Reached current target
            reward = 50
        else:
            reward = -1 * current_distance  # Negative reward based on distance
        
        # Add penalty for being far from trajectory
        if self.current_target_idx > 0 and self.current_target_idx < len(self.trajectory_points):
            # Calculate distance to line segment between previous and current target
            prev_point = self.trajectory_points[self.current_target_idx - 1]
            curr_point = self.trajectory_points[self.current_target_idx]
            
            # Simple distance calculation to line segment
            # This is a simplification - a proper implementation would calculate
            # the perpendicular distance to the line segment
            trajectory_penalty = self._distance_to_line_segment(
                self.state_space['end_effector_x'], 
                self.state_space['end_effector_y'],
                prev_point[0], prev_point[1],
                curr_point[0], curr_point[1]
            )
            
            # Apply penalty if too far from trajectory
            if trajectory_penalty > 10:
                reward -= trajectory_penalty * 0.5
        
        # Big reward for completing the full trajectory
        if self.trajectory_completed:
            reward += 500
            
        return reward
    
    def _distance_to_line_segment(self, x, y, x1, y1, x2, y2):
        """Calculate distance from point (x,y) to line segment (x1,y1)-(x2,y2)"""
        # Vector from line start to point
        dx = x - x1
        dy = y - y1
        
        # Vector representing the line segment
        line_dx = x2 - x1
        line_dy = y2 - y1
        
        # Line segment length squared
        line_length_sq = line_dx**2 + line_dy**2
        
        # If line segment is a point, return distance to that point
        if line_length_sq == 0:
            return math.sqrt(dx**2 + dy**2)
        
        # Calculate projection of point onto line
        t = max(0, min(1, (dx * line_dx + dy * line_dy) / line_length_sq))
        
        # Calculate closest point on line segment
        proj_x = x1 + t * line_dx
        proj_y = y1 + t * line_dy
        
        # Return distance to closest point
        return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
            
    def _is_terminal(self):
        """Check if current state is terminal"""
        # Terminal conditions:
        # 1. Completed the trajectory
        # 2. Out of workspace bounds
        
        if self.trajectory_completed:
            return True
            
        # Check if out of bounds
        if not self._is_within_workspace():
            return True
            
        return False
    
    def _is_within_workspace(self):
        """Check if end effector is within workspace bounds"""
        x = self.state_space['end_effector_x']
        y = self.state_space['end_effector_y']
        
        return (self.workspace_bounds[0] <= x <= self.workspace_bounds[1] and
                self.workspace_bounds[0] <= y <= self.workspace_bounds[1])
        
    def get_state(self):
        """Return current state as observation"""
        return [
            self.state_space['end_effector_x'],
            self.state_space['end_effector_y'],
            self.state_space['target_x'],
            self.state_space['target_y']
        ]

import pickle

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """Initialize Q-learning agent with hyperparameters"""
        self.env = env
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Rate at which epsilon decreases
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        
        # State discretization parameters
        self.state_bins = 20  # Number of bins for each state dimension
        self.state_bounds = {
            'end_effector_x': (-50, 50),
            'end_effector_y': (-50, 50),
            'target_x': (-50, 50),
            'target_y': (-50, 50)
        }
        
        # Initialize Q-table
        # 4D state space (end_effector_x, end_effector_y, target_x, target_y)
        # Each dimension is discretized into state_bins
        # For each state, we have len(env.actions) possible actions
        self.q_table = np.zeros((self.state_bins, self.state_bins, 
                                 self.state_bins, self.state_bins, 
                                 len(env.actions)))
        
        # Training statistics
        self.episode_rewards = []
        self.episode_steps = []
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state indices for Q-table"""
        # state = [end_effector_x, end_effector_y, target_x, target_y]
        discretized = []
        
        # Discretize each dimension
        dimensions = ['end_effector_x', 'end_effector_y', 'target_x', 'target_y']
        for i, dim in enumerate(dimensions):
            bounds = self.state_bounds[dim]
            # Clip value to bounds
            val = max(bounds[0], min(bounds[1], state[i]))
            # Scale to [0, 1]
            scaled = (val - bounds[0]) / (bounds[1] - bounds[0])
            # Convert to bin index
            bin_index = min(int(scaled * self.state_bins), self.state_bins - 1)
            discretized.append(bin_index)
        
        return tuple(discretized)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        discretized_state = self.discretize_state(state)
        
        # Exploration: choose random action
        if np.random.random() < self.epsilon:
            return np.random.choice(list(self.env.actions.keys()))
        
        # Exploitation: choose best action from Q-table
        return np.argmax(self.q_table[discretized_state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule"""
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discretized_state + (action,)]
        
        if done:
            # If terminal state, there is no next state
            target_q = reward
        else:
            # Max Q-value for next state
            max_next_q = np.max(self.q_table[discretized_next_state])
            # Q-learning update formula
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[discretized_state + (action,)] += self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, num_episodes=1000, max_steps=200, render=False, gui=None, metrics_logger=None):
        """Train the agent using Q-learning"""
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            # Episode loop
            for step in range(max_steps):
                # Choose action
                action = self.choose_action(state)
                
                # Take action
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                
                # Learn from experience
                self.learn(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                steps += 1
                
                # Render if requested
                if render and gui is not None:
                    gui.render(episode, step, reward, total_reward)
                    # Check for window close
                    if not gui.check_events():
                        return
                    # Small delay to make rendering visible
                    time.sleep(0.01)
                
                # Break if episode is done
                if done:
                    break
            
            # Log metrics at the end of each episode (not every step)
            if metrics_logger is not None:
                # Calculate average reward per step
                avg_reward = total_reward / max(1, steps)
                metrics_logger.log_episode(episode, steps, total_reward, avg_reward, self.epsilon, 
                                          self.env.current_target_idx, len(self.env.trajectory_points))
            
            # Decay epsilon after each episode
            self.decay_epsilon()
            
            # Store episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if episode > 0 else total_reward
                print(f"Episode {episode}/{num_episodes}, Steps: {steps}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        print("Training completed!")
        return self.q_table
    
    def save_q_table(self, filename="q_table.pkl"):
        """Save the Q-table to a file using pickle."""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        """Load the Q-table from a file using pickle."""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")

    def test(self, num_episodes=10, max_steps=200, render=True, gui=None, metrics_logger=None):
        """Test the trained agent"""
        total_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            # Episode loop
            for step in range(max_steps):
                # Choose best action (no exploration)
                discretized_state = self.discretize_state(state)
                action = np.argmax(self.q_table[discretized_state])
                
                # Take action
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                
                # Update state
                state = next_state
                steps += 1
                
                # Render if requested
                if render and gui is not None:
                    gui.render(episode, step, reward, total_reward)
                    # Check for window close
                    if not gui.check_events():
                        return
                    # Small delay to make rendering visible
                    time.sleep(0.05)
                
                # Break if episode is done
                if done:
                    if reward > 0:  # Positive reward means success
                        success_count += 1
                    break
            
            total_rewards.append(total_reward)
            print(f"Test Episode {episode+1}/{num_episodes}, Steps: {steps}, Total Reward: {total_reward:.2f}")
            
            # Log test episode metrics
            if metrics_logger is not None:
                metrics_logger.log_episode(episode, steps, total_reward, total_reward / max(1, steps), 0, 
                                          self.env.current_target_idx, len(self.env.trajectory_points))
        
        # Print test results
        avg_reward = np.mean(total_rewards)
        success_rate = success_count / num_episodes * 100
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")

class MetricsLogger:
    def __init__(self, output_dir="metrics"):
        """Initialize metrics logger"""
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize training metrics storage
        self.training_metrics = {
            "episode_rewards": [],
            "episode_steps": [],
            "avg_rewards": [],
            "epsilon_values": [],
            "waypoints_reached": [],
            "trajectory_completion": []
        }
        
        # Initialize testing metrics storage
        self.testing_metrics = {
            "episode_rewards": [],
            "episode_steps": [],
            "success_count": 0,
            "waypoints_reached": [],
            "trajectory_completion": []
        }
        
        # Training metadata
        self.metadata = {
            "training_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_end": None,
            "num_episodes": 0,
            "learning_rate": 0,
            "discount_factor": 0,
            "initial_epsilon": 0,
            "epsilon_decay": 0,
            "epsilon_min": 0
        }
        
        # Current mode (training or testing)
        self.current_mode = "training"
    
    def set_mode(self, mode):
        """Set current mode (training or testing)"""
        if mode in ["training", "testing"]:
            self.current_mode = mode
    
    def log_episode(self, episode, steps, reward, avg_reward, epsilon, waypoint, total_waypoints):
        """Log metrics for a single episode (training or testing)"""
        completion = waypoint / total_waypoints
        
        if self.current_mode == "training":
            # Log training metrics
            self.training_metrics["episode_rewards"].append(reward)
            self.training_metrics["episode_steps"].append(steps)
            self.training_metrics["avg_rewards"].append(avg_reward)
            self.training_metrics["epsilon_values"].append(epsilon)
            self.training_metrics["waypoints_reached"].append(waypoint)
            self.training_metrics["trajectory_completion"].append(completion)
        else:
            # Log testing metrics
            self.testing_metrics["episode_rewards"].append(reward)
            self.testing_metrics["episode_steps"].append(steps)
            self.testing_metrics["waypoints_reached"].append(waypoint)
            self.testing_metrics["trajectory_completion"].append(completion)
            
            # Count successful episodes (completed trajectory)
            if completion >= 0.99:  # Consider 99% completion as success
                self.testing_metrics["success_count"] += 1
    
    def update_metadata(self, agent, num_episodes):
        """Update metadata with agent parameters"""
        self.metadata["num_episodes"] = num_episodes
        self.metadata["learning_rate"] = agent.learning_rate
        self.metadata["discount_factor"] = agent.discount_factor
        self.metadata["initial_epsilon"] = agent.epsilon
        self.metadata["epsilon_decay"] = agent.epsilon_decay
        self.metadata["epsilon_min"] = agent.epsilon_min
        self.metadata["training_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_metrics(self):
        """Save all metrics to separate files for training and testing"""
        # Save training metrics to CSV
        training_csv = os.path.join(self.output_dir, f"training_metrics_{self.timestamp}.csv")
        with open(training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Steps', 'Reward', 'Avg_Reward', 'Epsilon', 'Waypoints', 'Completion'])
            for i in range(len(self.training_metrics["episode_rewards"])):
                writer.writerow([
                    i, 
                    self.training_metrics["episode_steps"][i], 
                    self.training_metrics["episode_rewards"][i], 
                    self.training_metrics["avg_rewards"][i], 
                    self.training_metrics["epsilon_values"][i],
                    self.training_metrics["waypoints_reached"][i],
                    self.training_metrics["trajectory_completion"][i]
                ])
        
        # Save testing metrics to CSV
        testing_csv = os.path.join(self.output_dir, f"testing_metrics_{self.timestamp}.csv")
        with open(testing_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Steps', 'Reward', 'Waypoints', 'Completion'])
            for i in range(len(self.testing_metrics["episode_rewards"])):
                writer.writerow([
                    i, 
                    self.testing_metrics["episode_steps"][i], 
                    self.testing_metrics["episode_rewards"][i],
                    self.testing_metrics["waypoints_reached"][i],
                    self.testing_metrics["trajectory_completion"][i]
                ])
        
        # Save test summary to JSON
        test_summary = {
            "avg_reward": np.mean(self.testing_metrics["episode_rewards"]) if self.testing_metrics["episode_rewards"] else 0,
            "success_rate": (self.testing_metrics["success_count"] / len(self.testing_metrics["episode_rewards"]) * 100) 
                            if self.testing_metrics["episode_rewards"] else 0,
            "avg_steps": np.mean(self.testing_metrics["episode_steps"]) if self.testing_metrics["episode_steps"] else 0,
            "num_episodes": len(self.testing_metrics["episode_rewards"])
        }
        
        test_summary_file = os.path.join(self.output_dir, f"test_summary_{self.timestamp}.json")
        with open(test_summary_file, 'w') as f:
            json.dump(test_summary, f, indent=4)
        
        # Save metadata to JSON
        meta_file = os.path.join(self.output_dir, f"metadata_{self.timestamp}.json")
        with open(meta_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        print(f"Training metrics saved to {training_csv}")
        print(f"Testing metrics saved to {testing_csv}")
        print(f"Test summary saved to {test_summary_file}")
        print(f"Metadata saved to {meta_file}")
        
        return training_csv, testing_csv
    
    def plot_metrics(self):
        """Generate and save separate plots for training and testing metrics"""
        # Create training plots
        fig1, axs1 = plt.subplots(2, 2, figsize=(15, 10))
        fig1.suptitle('Training Metrics', fontsize=16)
        
        # Plot episode rewards
        axs1[0, 0].plot(self.training_metrics["episode_rewards"])
        axs1[0, 0].set_title('Episode Rewards')
        axs1[0, 0].set_xlabel('Episode')
        axs1[0, 0].set_ylabel('Total Reward')
        axs1[0, 0].grid(True)
        
        # Plot average rewards
        axs1[0, 1].plot(self.training_metrics["avg_rewards"])
        axs1[0, 1].set_title('Average Rewards Per Step')
        axs1[0, 1].set_xlabel('Episode')
        axs1[0, 1].set_ylabel('Average Reward')
        axs1[0, 1].grid(True)
        
        # Plot epsilon decay
        axs1[1, 0].plot(self.training_metrics["epsilon_values"])
        axs1[1, 0].set_title('Epsilon Decay')
        axs1[1, 0].set_xlabel('Episode')
        axs1[1, 0].set_ylabel('Epsilon')
        axs1[1, 0].grid(True)
        
        # Plot trajectory completion
        axs1[1, 1].plot(self.training_metrics["trajectory_completion"])
        axs1[1, 1].set_title('Trajectory Completion')
        axs1[1, 1].set_xlabel('Episode')
        axs1[1, 1].set_ylabel('Completion Percentage')
        axs1[1, 1].grid(True)
        
        # Adjust layout and save training plots
        plt.tight_layout()
        training_plot_file = os.path.join(self.output_dir, f"training_curves_{self.timestamp}.png")
        plt.savefig(training_plot_file)
        plt.close()
        
        # Create testing plots if we have testing data
        if self.testing_metrics["episode_rewards"]:
            fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
            fig2.suptitle('Testing Metrics', fontsize=16)
            
            # Plot episode rewards
            axs2[0, 0].plot(self.testing_metrics["episode_rewards"])
            axs2[0, 0].set_title('Test Episode Rewards')
            axs2[0, 0].set_xlabel('Episode')
            axs2[0, 0].set_ylabel('Total Reward')
            axs2[0, 0].grid(True)
            
            # Plot episode steps
            axs2[0, 1].plot(self.testing_metrics["episode_steps"])
            axs2[0, 1].set_title('Test Episode Steps')
            axs2[0, 1].set_xlabel('Episode')
            axs2[0, 1].set_ylabel('Steps')
            axs2[0, 1].grid(True)
            
            # Plot trajectory completion
            axs2[1, 0].plot(self.testing_metrics["trajectory_completion"])
            axs2[1, 0].set_title('Test Trajectory Completion')
            axs2[1, 0].set_xlabel('Episode')
            axs2[1, 0].set_ylabel('Completion Percentage')
            axs2[1, 0].grid(True)
            
            # Plot waypoints reached
            axs2[1, 1].plot(self.testing_metrics["waypoints_reached"])
            axs2[1, 1].set_title('Test Waypoints Reached')
            axs2[1, 1].set_xlabel('Episode')
            axs2[1, 1].set_ylabel('Waypoints')
            axs2[1, 1].grid(True)
            
            # Adjust layout and save testing plots
            plt.tight_layout()
            testing_plot_file = os.path.join(self.output_dir, f"testing_curves_{self.timestamp}.png")
            plt.savefig(testing_plot_file)
            plt.close()
            
            print(f"Training curves saved to {training_plot_file}")
            print(f"Testing curves saved to {testing_plot_file}")
            return training_plot_file, testing_plot_file
        else:
            print(f"Training curves saved to {training_plot_file}")
            return training_plot_file, None

# Main function to run Q-learning training and testing
def train_and_test_q_learning():
    # Create environment
    env = PULSR2Environment()
    
    # Create GUI
    gui = PULSR2GUI(env)
    
    # Create metrics logger
    metrics_logger = MetricsLogger()
    
    # Create Q-learning agent with optimized parameters for faster learning
    agent = QLearningAgent(
        env,
        learning_rate=0.2,         # Increased from 0.1 for faster learning
        discount_factor=0.95,      # Slightly increased to value future rewards more
        epsilon=0.5,               # Start with less exploration (was 1.0)
        epsilon_decay=0.9,         # Faster decay (was 0.995)
        epsilon_min=0.01           # Same minimum exploration
    )
    
    # Update metadata
    metrics_logger.update_metadata(agent, 100)
    
    # Train agent with fewer episodes
    print("Starting training...")
    agent.train(num_episodes=100, max_steps=200, render=True, gui=gui, metrics_logger=metrics_logger)
    
    # Save metrics
    metrics_logger.save_metrics()

    # Save Q-table after training
    agent.save_q_table("q_table.pkl")
    
    # Plot metrics
    metrics_logger.plot_metrics()
    
    # Test agent
    print("Starting testing...")
    metrics_logger.set_mode("testing")
    agent.test(num_episodes=10, max_steps=200, render=True, gui=gui, metrics_logger=metrics_logger)
    
    # Close GUI
    gui.close()

# Update the main function to include options for manual control or Q-learning
def main():
    print("Select mode:")
    print("1. Manual Control")
    print("2. Q-Learning Training and Testing")
    
    # Automatically select Q-Learning mode for demonstration
    mode = "2"  # Change this to "1" for manual control
    print(f"Selected mode: {mode}")
    
    if mode == "1":
        # Manual control mode
        manual_control()
    elif mode == "2":
        # Q-learning mode
        train_and_test_q_learning()
    else:
        print("Invalid mode. Defaulting to manual control.")
        manual_control()

# Function for manual control mode
def manual_control():
    # Create environment
    env = PULSR2Environment()
    
    # Create GUI
    gui = PULSR2GUI(env)
    
    # Initialize variables
    episode = 0
    step = 0
    total_reward = 0
    current_reward = 0
    running = True
    
    # Main loop
    while running:
        # Check for pygame events
        running = gui.check_events()
        
        # Handle keyboard input for manual control
        keys = pygame.key.get_pressed()
        action = None
        
        if keys[pygame.K_UP]:
            action = 0  # North
        elif keys[pygame.K_DOWN]:
            action = 1  # South
        elif keys[pygame.K_RIGHT]:
            action = 2  # East
        elif keys[pygame.K_LEFT]:
            action = 3  # West
        elif keys[pygame.K_r]:
            # Reset environment
            env.reset()
            episode += 1
            step = 0
            total_reward = 0
        
        # Take action if one was selected
        if action is not None:
            next_state, reward, done = env.step(action)
            current_reward = reward
            total_reward += reward
            step += 1
            
            # If episode is done, reset
            if done:
                print(f"Episode {episode} completed in {step} steps with total reward {total_reward}")
                env.reset()
                episode += 1
                step = 0
                total_reward = 0
        
        # Render the environment
        gui.render(episode, step, current_reward, total_reward)
        
        # Small delay to control speed
        time.sleep(0.1)
    
    # Close GUI when done
    gui.close()

# Run the main function when script is executed
if __name__ == "__main__":
    main()