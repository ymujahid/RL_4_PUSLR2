import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple

class PulsrEnv(gym.Env):
    def __init__(
        self,
        workspace_csv: Optional[str] = None,
        trajectory_csv: Optional[str] = None,
        max_step: float = 1.0,
        discrete_actions: bool = True,
    ):
        """
        Continuous x-y plane environment initializer.

        Args:
            workspace_csv: Path to CSV containing allowed workspace points (columns: x,y).
                           If None, a default square [-10,10]x[-10,10] workspace is used.
            trajectory_csv: Path to CSV containing ordered target trajectory points (columns: x,y).
                            If None, an empty trajectory is created.
            max_step: maximum step size (L-inf bound) for continuous delta actions.
            discrete_actions: when True, action_space is Discrete(4) mapped to N/E/S/W with step size = max_step.
                              when False (default), action_space is Box([-max_step,-max_step],[max_step,max_step]) meaning Δx,Δy.
        """
        super().__init__()

        self.max_step = float(max_step)
        self.discrete_actions = bool(discrete_actions)

        # --- Load workspace points (explicit set of allowed x,y coordinates) ---
        if workspace_csv:
            df_ws = pd.read_csv(workspace_csv)
            # try to find x,y columns; fall back to first two numeric columns
            if {"x", "y"}.issubset(df_ws.columns.str.lower()):
                # handle possible mixed-case headers by case-insensitive match
                cols = {c.lower(): c for c in df_ws.columns}
                xcol = cols['x']; ycol = cols['y']
                pts = df_ws[[xcol, ycol]].to_numpy(dtype=float)
            else:
                # use first two numeric columns
                numeric_cols = df_ws.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    raise ValueError("workspace_csv must contain at least two numeric columns for x and y")
                pts = df_ws[numeric_cols[:2]].to_numpy(dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 2:
                raise ValueError("workspace CSV must have two columns (x,y)")
            self.workspace_points = pts
        else:
            # Default workspace: dense grid corners of [-10,10] square (used only to determine bounds)
            xs = np.linspace(-10.0, 10.0, 51)
            ys = np.linspace(-10.0, 10.0, 51)
            xv, yv = np.meshgrid(xs, ys)
            self.workspace_points = np.column_stack([xv.ravel(), yv.ravel()])

        # Compute workspace bounds from workspace_points
        self.x_min, self.y_min = np.min(self.workspace_points, axis=0)
        self.x_max, self.y_max = np.max(self.workspace_points, axis=0)

        # --- Load target trajectory (ordered list of x,y coordinates) ---
        if trajectory_csv:
            df_traj = pd.read_csv(trajectory_csv)
            # similar detection as workspace
            if {"x", "y"}.issubset(df_traj.columns.str.lower()):
                cols = {c.lower(): c for c in df_traj.columns}
                xcol = cols['x']; ycol = cols['y']
                traj = df_traj[[xcol, ycol]].to_numpy(dtype=float)
            else:
                numeric_cols = df_traj.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    raise ValueError("trajectory_csv must contain at least two numeric columns for x and y")
                traj = df_traj[numeric_cols[:2]].to_numpy(dtype=float)
            if traj.ndim != 2 or traj.shape[1] != 2:
                raise ValueError("trajectory CSV must have two columns (x,y)")
            self.trajectory = traj
        else:
            # default empty trajectory: a fixed single point at top middle of workspace
            mid_x = 0.5 * (self.x_min + self.x_max)
            self.trajectory = np.array([[mid_x, self.y_max]], dtype=float)

        self.trajectory_length = len(self.trajectory)
        self.trajectory_index = 0  # current index into trajectory

        # --- Observation space: [agent_x, agent_y, target_x, target_y, distance] ---
        obs_low = np.array([self.x_min, self.y_min, self.x_min, self.y_min, 0.0], dtype=np.float32)
        obs_high = np.array([self.x_max, self.y_max, self.x_max, self.y_max,
                             np.sqrt((self.x_max - self.x_min) ** 2 + (self.y_max - self.y_min) ** 2)],
                            dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)


        # --- Action space ---
        if self.discrete_actions:
            # 4-directional discrete actions mapped to Δx,Δy of magnitude max_step
            self.action_space = spaces.Discrete(4)
            self._discrete_to_delta = {
                0: np.array([0.0, self.max_step]),   # North (increase y)
                1: np.array([self.max_step, 0.0]),   # East  (increase x)
                2: np.array([0.0, -self.max_step]),  # South (decrease y)
                3: np.array([-self.max_step, 0.0]),  # West  (decrease x)
            }
        else:
            # Continuous action: delta x, delta y in [-max_step, max_step]
            self.action_space = spaces.Box(
                low=np.array([-self.max_step, -self.max_step], dtype=np.float32),
                high=np.array([self.max_step, self.max_step], dtype=np.float32),
                dtype=np.float32
            )

        # --- Internal agent & target state (floats) ---
        self._agent_location = np.array([0.0, 0.0], dtype=float)    # will be set in reset()
        self._target_location = np.array([0.0, 0.0], dtype=float)   # updated from trajectory

        # initialize environment (sets start positions)
        self.reset()

    # Example helper to get observation in required dict form
    def _get_obs(self):
        """
        Gets the x-y coordinates of the agent and the target, and the euclidean distance between them as the observation

        """
        ax, ay = self._agent_location
        tx, ty = self._target_location
        dist = np.linalg.norm([ax - tx, ay - ty])
        return np.array([ax, ay, tx, ty, dist], dtype=np.float32)

    # Note: implement reset() and step() according to how you want the agent to be placed & trajectory advanced.
    # Example reset (random agent position sampled from workspace_points and target at trajectory start):
    
    def reset(self, *, seed: int = None, options: dict = None):
        """Reset the environment to an initial state.
    
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Custom reset options.
    
        Returns:
            observation (np.ndarray): Initial observation.
            info (dict): Additional info.
        """
        # Handle seeding
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
    
        # Pick agent start location
        idx = np.random.randint(len(self.workspace_points)) # collect the list of the start points of all data samples and put them here to speed up training
        self._agent_location = self.workspace_points[idx].astype(float).copy()
    
        # Reset target to first trajectory point
        self.trajectory_index = 0
        self._target_location = self.trajectory[self.trajectory_index].astype(float).copy()
    
        # Return observation and info (Gymnasium requires both)
        return self._get_obs(), {}

    def step(self, action):
        """Execute one timestep within the continuous XY environment.
    
        Args:
            action: 
                - If discrete_actions=True → int in {0,1,2,3} mapped to N,E,S,W.
                - If discrete_actions=False → np.array([dx, dy]) continuous delta.
    
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # --- Apply action ---
        if self.discrete_actions:
            delta = self._discrete_to_delta[int(action)]
        else:
            # continuous action, clip to allowed max_step
            delta = np.clip(action, -self.max_step, self.max_step)
    
        # Update agent position and clamp to workspace bounds
        self._agent_location = np.clip(
            self._agent_location + delta,
            [self.x_min, self.y_min],
            [self.x_max, self.y_max]
        )
    
        # --- Advance target along trajectory ---
        self.trajectory_index = min(self.trajectory_index + 1, self.trajectory_length - 1)
        self._target_location = self.trajectory[self.trajectory_index].astype(float).copy()
    
        # --- Check termination: agent exactly on target (rare in continuous) ---
        # terminated = np.allclose(self._agent_location, self._target_location, atol=1e-3)

        terminated = False
    
        # You can also terminate if trajectory is finished
        if self.trajectory_index >= self.trajectory_length - 1:
            terminated = True
    
        # --- Truncation (optional: step limit) ---
        truncated = False
    
        # --- Reward ---
        # Here: negative Euclidean distance (closer to target = higher reward)
        dist = np.linalg.norm(self._agent_location - self._target_location)
        if dist <= 10:
            reward = 1
        else:
            reward = -1
    
        # --- Build observation & info ---
        observation = self._get_obs()
        info = {"distance": dist}
    
        return observation, reward, terminated, truncated, info














# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium.envs.registration import register
# from enum import Enum
# import numpy as np

# # Register this module as a gym environment. Once registered, the id is usable in gym.make().
# # When running this code, you can ignore this warning: "UserWarning: WARN: Overriding environment airplane-boarding-v0 already in registry."
# register(
#     id='pulsr-virtualenv-v0',
#     entry_point='pulsrEnv:PulsrEnv', # module_name:class_name
# )

# class PulsrEnv(gym.Env):
#     metadata = {'render_modes': ['human']}
#     def __init__(self, render_mode=None):
#         # Actions: 0=N, 1=E, 2=S, 3=W
#         self.action_space = Discrete(4)
        
#         # State: [x, y, score]
#         self.observation_space = Box(
#             low=np.array([-100.0, -100.0, 0.0], dtype=np.float32),
#             high=np.array([100.0, 100.0, np.inf], dtype=np.float32),
#             dtype=np.float32
#         )
        
#         # Circle radius for the ball trajectory
#         self.radius = 100
#         self.max_steps = 239
        
#         # Visualization setup
#         self.fig, self.ax = None, None
#         self.agent_plot, self.ball_plot, self.traj_plot = None, None, None
#         self.trajectory = []
        
#         self.reset()
    
#     def step(self, action):
#         old_x, old_y, _ = self.state
        
#         # --- Apply action ---
#         if action == 0:   # North
#             self.state[1] += 1
#         elif action == 1: # East
#             self.state[0] += 1
#         elif action == 2: # South
#             self.state[1] -= 1
#         elif action == 3: # West
#             self.state[0] -= 1
        
#         # --- Keep agent inside workspace ---
#         if not (-100 <= self.state[0] <= 100 and -100 <= self.state[1] <= 100):
#             self.state[0], self.state[1] = old_x, old_y
        
#         # --- Move ball along circular trajectory ---
#         angle = (2 * np.pi * self.episode_length) / self.max_steps
#         self.ball_pos = np.array([
#             self.radius * np.sin(angle),   # x
#             self.radius * np.cos(angle)    # y (starts at top)
#         ])
        
#         # --- Distance & scoring ---
#         d = np.linalg.norm(self.state[:2] - self.ball_pos)
#         if d <= 70:
#             self.state[2] += 1   # increase score
        
#         # --- Use score as reward ---
#         reward = self.state[2]
        
#         # --- Step update ---
#         self.episode_length += 1
#         done = self.episode_length >= self.max_steps
        
#         # --- Final reward mapping ---
#         info = {}
#         if done:
#             score = self.state[2]
#             if score > 150:
#                 info["final_reward"] = 3
#             elif 100 <= score <= 150:
#                 info["final_reward"] = 2
#             elif 90 <= score < 100:
#                 info["final_reward"] = 1
#             elif score == 90:
#                 info["final_reward"] = 0
#             elif 70 <= score < 90:
#                 info["final_reward"] = -1
#             elif 50 <= score < 70:
#                 info["final_reward"] = -2
#             else:
#                 info["final_reward"] = -3
        
#         return self.state.copy(), reward, done, info
        
#     def reset(self):
#         # Reset agent and trajectory
#         self.state = np.array([0, 0, 0], dtype=np.float32)
#         self.episode_length = 0
#         self.trajectory = []
        
#         # Start ball at angle=0
#         self.ball_pos = np.array([self.radius, 0])
        
#         return self.state.copy()
    
#     def render(self, mode="human"):
#         if self.fig is None:
#             plt.ion()
#             self.fig, self.ax = plt.subplots(figsize=(6, 6));
#             # plt.close(self.fig)   # suppress <Figure ...> in notebook output
            
#             self.ax.set_xlim(-120, 120)
#             self.ax.set_ylim(-120, 120)
#             self.ax.set_aspect("equal")
#             self.ax.set_title("PULSR Environment")
            
#             circle = plt.Circle((0, 0), self.radius, color="gray", fill=False, linestyle="--")
#             self.ax.add_patch(circle)
            
#             self.agent_plot, = self.ax.plot([], [], "bo", label="Agent")
#             self.ball_plot, = self.ax.plot([], [], "ro", label="Ball")
#             self.traj_plot, = self.ax.plot([], [], "b--", linewidth=0.8, alpha=0.6, label="Trajectory")
            
#             self.ax.legend()

#         # Update agent and ball positions
#         self.agent_plot.set_data([self.state[0]], [self.state[1]])
#         self.ball_plot.set_data([self.ball_pos[0]], [self.ball_pos[1]])
        
#         if len(self.trajectory) > 1:
#             traj = np.array(self.trajectory)
#             self.traj_plot.set_data(traj[:, 0], traj[:, 1])
        
#         plt.draw()
#         plt.pause(0.01)
    
#     #check validity of the environment

#     def my_check_env():
#         from gymnasium.utils.env_checker import check_env
#         env = gym.make('pulsr-virtualenv-v0', render_mode='human')
#         check_env(env.unwrapped)
    
#     if __name__ == "__main__":
#         my_check_env()
