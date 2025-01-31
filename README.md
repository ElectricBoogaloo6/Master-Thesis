# "Guide Me Through the Unexpected": Investigating How Deviation From Expectation Affects Human Teaching And Robot Learning.​

This project explores how deviations from expectations influence human teaching strategies and robot learning efficiency. Using a custom robotic task environment built in PyBullet, the study examines how exposing a robot's internal state information affects human decision-making when selecting demonstrations. The research involves both a user study and a simulation study, where we analyzed two teaching strategies: result-oriented and expectation-oriented, to determine their impact on robot performance. Key findings provide insights into optimizing human-robot interaction and aligning human teaching strategies with reinforcement learning algorithms. This research was conducted under the supervision of [Dr.ir. Kim Baraka](https://research.vu.nl/en/persons/kim-baraka) and [Muhan Hou](https://research.vu.nl/en/persons/muhan-hou).

Link for the research paper will be added here in the future. 

# The Robot Task

This project involves a robotic pushing task where a robotic arm interacts with objects in a simulated environment. The task setup includes:
- **Goal**: The objective is to push a green cube toward a designated goal position, represented by a translucent green cube.
- **Starting Positions**: The colored sticks indicate 10 possible starting positions for the green cube.
- **Obstacle**: A red bar is placed in the middle, representing an obstacle that the robot must navigate around.
- **Episodic Interaction**: Participants observe the full rollout of each attempt.
<p align="center">
  <img src="https://github.com/user-attachments/assets/bc912bcc-844d-498c-88f2-9bf63071336b" width="400">
</p>

# Visualisation

During the user study, we help participants understand the robot's movement and learning process via visual feedback:
- **Colored Lines**: These represent the trajectory of the object as it moves, serving as a history of past rollouts.
- **White Line**: This indicates the movement of the robot’s actuator during each rollout, helping participants track the robot's movement.
- **Participant Feedback on Visual Overload**:
  - 9 participants found the visual information comfortable.
  - 1 participant had a neutral opinion.
  - 2 participants found it slightly overwhelming.

This visualization approach was used to balance clarity and cognitive load while ensuring participants could make informed decisions.
<p align="center">
  <img src="https://github.com/user-attachments/assets/6bfbb28e-a0f2-42cd-859d-6be08eff132c" width="400">
</p>

# Trained models for each task difficulty can be found and downloaded from [here](https://drive.google.com/drive/folders/1hXYEw3sfe5ofgur2Y6SPbylyvQJjw_nD?usp=drive_link).




