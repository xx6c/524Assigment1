import numpy as np
import tkinter as tk
import random
from tkinter import messagebox
import time

# Define the mesh size and the square size
grid_size = 10
square_size = 50

# Create a game window
root = tk.Tk()
root.title("Grid world game")

# Create canvas
canvas = tk.Canvas(root, width=grid_size * square_size, height=grid_size * square_size)
canvas.pack()

# Randomly generate obstacles and traps
obstacles = []
traps = []
num_obstacles = 5
num_traps = 3
while len(obstacles) < num_obstacles:
    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    if (x, y) != (0, 0) and (x, y) != (grid_size - 1, grid_size - 1):
        obstacles.append((x, y))

while len(traps) < num_traps:
    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    if (x, y) not in obstacles and (x, y) != (0, 0) and (x, y) != (grid_size - 1, grid_size - 1):
        traps.append((x, y))

# Randomly generate item positions
items = []
num_items = 6
while len(items) < num_items:
    x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
    if (x, y) not in obstacles and (x, y) not in traps and (x, y) != (0, 0) and (x, y) != (grid_size - 1, grid_size - 1):
        items.append((x, y))

# paint grid
def draw_grid():
    for i in range(grid_size):
        for j in range(grid_size):
            canvas.create_rectangle(i * square_size, j * square_size, (i + 1) * square_size, (j + 1) * square_size,
                                    outline='gray')

# paint agent
agent = None
def draw_agent(state):
    global agent
    if agent:
        canvas.delete(agent)
    x, y = state
    agent = canvas.create_oval(x * square_size + 10, y * square_size + 10, x * square_size + 40, y * square_size + 40,
                               fill='blue')

# paint end
end = None
def draw_end():
    global end
    end_state = (grid_size - 1, grid_size - 1)
    if end:
        canvas.delete(end)
    x, y = end_state
    end = canvas.create_rectangle(x * square_size + 10, y * square_size + 10, x * square_size + 40, y * square_size + 40,
                                  fill='green')

# Plot obstacles
def draw_obstacles():
    for obstacle in obstacles:
        x, y = obstacle
        canvas.create_rectangle(x * square_size + 10, y * square_size + 10, x * square_size + 40, y * square_size + 40,
                                fill='red')

# Draw trap
def draw_traps():
    for trap in traps:
        x, y = trap
        canvas.create_rectangle(x * square_size + 10, y * square_size + 10, x * square_size + 40, y * square_size + 40,
                                fill='black')

# Draw item
item_objects = []
def draw_items():
    global item_objects
    for item_obj in item_objects:
        canvas.delete(item_obj)
    item_objects = []
    for item in items:
        x, y = item
        item_obj = canvas.create_oval(x * square_size + 20, y * square_size + 20, x * square_size + 30, y * square_size + 30,
                                      fill='yellow')
        item_objects.append(item_obj)

# Initialize the Q table
q_table = np.zeros((grid_size, grid_size, 4))

# Q - learning update
def q_learning_update(state, action, reward, next_state, learning_rate, discount_factor):
    predict = q_table[state[0], state[1], action]
    target = reward + discount_factor * np.max(q_table[next_state[0], next_state[1], :])
    q_table[state[0], state[1], action] = (1 - learning_rate) * predict + learning_rate * target
    return q_table

# epsilon - greedy
def epsilon_greedy(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(q_table[state[0], state[1], :])

# Calculating Manhattan distance
def manhattan_distance(state1, state2):
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

# Initializes the proxy state
state = (0, 0)
step_count = 0
max_steps = 500
has_item = False

learning_rate = 0.15
discount_factor = 0.95
epsilon = 0.95
epsilon_decay = 0.995
min_epsilon = 0.001

def run_episode():
    global state, step_count, has_item, epsilon, recent_states
    state = (0, 0)
    step_count = 0
    has_item = False
    items = []
    num_items = 6
    while len(items) < num_items:
        x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        if (x, y) not in obstacles and (x, y) not in traps and (x, y) != (0, 0) and (x, y) != (grid_size - 1, grid_size - 1):
            items.append((x, y))
    draw_agent(state)
    draw_items()
    recent_states = []
    epsilon = 0.95

    while step_count < max_steps:
        step_count += 1
        action = epsilon_greedy(state, epsilon)
        new_x, new_y = state
        if action == 0 and new_y > 0:
            if (new_x, new_y - 1) in obstacles and not has_item:
                new_x, new_y = state
            else:
                new_y -= 1
        elif action == 1 and new_y < grid_size - 1:
            if (new_x, new_y + 1) in obstacles and not has_item:
                new_x, new_y = state
            else:
                new_y += 1
        elif action == 2 and new_x > 0:
            if (new_x - 1, new_y) in obstacles and not has_item:
                new_x, new_y = state
            else:
                new_x -= 1
        elif action == 3 and new_x < grid_size - 1:
            if (new_x + 1, new_y) in obstacles and not has_item:
                new_x, new_y = state
            else:
                new_x += 1
        next_state = (new_x, new_y)

        end_state = (grid_size - 1, grid_size - 1)
        dist_current_to_end = manhattan_distance(state, end_state)
        dist_next_to_end = manhattan_distance(next_state, end_state)

        reward = -1 * step_count
        if dist_next_to_end < dist_current_to_end:
            reward += 15 * (dist_current_to_end - dist_next_to_end)

        if next_state in items:
            items.remove(next_state)
            draw_items()
            reward += 20
            has_item = True
            print("Pick up props, temporary through the wall!")
        elif next_state in traps:
            reward -= 30
            print("Step into a trap, get punished!")
        elif next_state == end_state:
            reward += 100
            messagebox.showinfo("Game over", "Congratulations, you made it to the finish line!")
            break

        if next_state in recent_states:
            reward -= 5

        recent_states.append(next_state)
        if len(recent_states) > 5:
            recent_states.pop(0)

        q_table = q_learning_update(state, action, reward, next_state, learning_rate, discount_factor)

        state = next_state
        draw_agent(state)
        draw_items()
        print(f"Current Step: {step_count}, Current Reward: {reward}")

        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        root.update()
        time.sleep(0.3)


# Initialization interface
draw_grid()
draw_agent(state)
draw_end()
draw_obstacles()
draw_traps()
draw_items()

num_episodes = 1000
for episode in range(num_episodes):
    run_episode()

root.mainloop()