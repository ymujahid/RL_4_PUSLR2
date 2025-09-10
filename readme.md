# Reinforcement Learning Control of PULSR2 Rehabilitation Robot

This project develops a **Reinforcement Learning (RL) control system** for the **PULSR2 robot**—a brushless DC motor–actuated parallelogram arm designed for upper-limb stroke rehabilitation.  
The goal is to replace rigid, pre-programmed controllers with **adaptive, patient-specific control strategies** powered by reinforcement learning.

---

## 🚀 Project Overview

Stroke rehabilitation requires intensive and repetitive training. While conventional rehabilitation robots provide consistent therapy, their controllers cannot adapt to individual recovery patterns.  
This project addresses that limitation by:

- Modeling the kinematics of the **PULSR2 robot**.
- Designing a **custom RL environment** following **Gymnasium standards**.
- Implementing and training **Deep Q-Networks (DQN)** and Q-Learning agents.
- Evaluating trajectory-tracking performance through simulation.
- Preparing the groundwork for integration with real-world robotic hardware.

---

## 🧠 Key Features

- **Custom Gymnasium Environment**: Simulates the kinematics of the PULSR2 robot with defined state, action, and reward spaces.
- **Reinforcement Learning Algorithms**:
  - Deep Q-Network (DQN) benchmark model.
  - Classical Q-Learning baseline.
- **Data Processing Pipeline**:
  - Converts raw episode logs into structured CSVs.
  - Extracts end-effector trajectories and reward signals.
- **Visualization Tools**:
  - End-effector path plotting.
  - Reward distribution and training progress charts.
- **Experiment Tracking**:
  - Organized artifacts, models, and plots for reproducibility.

---

---

## ⚙️ Methodology

1. **Kinematic Modeling**  
   - Derived state variables: end-effector position, target position, and tracking error.  
   - Defined discrete action space (North, East, South, West).  
   - Designed a reward function based on proximity to the target trajectory.  

2. **Environment Development**  
   - Built custom environment (`pulsrEnv.py`) following Gymnasium standards.  
   - Validated with `check_env` for compatibility.  

3. **Training & Evaluation**  
   - Trained a **Deep Q-Network** with two hidden layers (64 ReLU neurons each).  
   - Benchmarked performance over 100,000 timesteps, achieving an **average episode reward of 230.5**.  
   - Compared results against baseline Q-Learning.  

4. **Visualization & Analysis**  
   - Plotted trajectory paths and reward curves using **Matplotlib**.  
   - Analyzed agent exploration vs. exploitation behavior.  

---

## 🛠️ Tools & Technologies

- **Programming Language**: Python 3.10+
- **Libraries & Frameworks**:
  - [Gymnasium](https://gymnasium.farama.org/) – RL environment standard
  - [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) – RL algorithms (DQN)
  - NumPy, Pandas – Data processing
  - Matplotlib, Seaborn – Visualization
- **Development Environment**: Google Colab / Jupyter Notebook
- **Version Control**: GitHub

---

## 📊 Results

- **DQN agent** successfully learned effective trajectory-tracking strategies.  
- Achieved **average reward ~230.5 per episode** after 100k timesteps.  
- Demonstrated the feasibility of **adaptive RL-based control** for rehabilitation robots.  
- Established groundwork for **hardware integration** with Raspberry Pi and motor sensors.  

---

## 🔮 Future Work

- Incorporate **robot dynamics** alongside kinematics for more realistic modeling.  
- Bridge C# control code with Python RL models (via API or codebase standardization).  
- Upgrade force sensors to enable **assist-as-needed** control modes.  
- Extend to **clinical trials** for patient-specific rehabilitation.  

---

## 📖 References

This work is based on the undergraduate thesis:  
**“Development of Reinforcement Learning Control of a Brushless DC Motor Actuated Parallelogram Arm Rehabilitation Robot”**  
by *Yunus Mujahid Olalekan*, Obafemi Awolowo University, 2025.  

---

## 👨‍💻 Author

**Mujahid Yunus**  
- Machine Learning Developer | Robotics Enthusiast  
- [Portfolio](https://mujahid4mldev.netlify.app/)  
- [GitHub](https://github.com/ymujahid)  


