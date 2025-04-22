import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib

# 設定 Matplotlib 字型
matplotlib.rc("font", family="Microsoft JhengHei")

# =============== 生成隨機迷宮 ===============
def ran_maze(rows, cols, obstacle_ratio):
    maze = np.zeros((rows, cols), dtype=int)
    num_obstacles = int(rows * cols * min(obstacle_ratio, 0.4))  # 限制最多 40% 障礙物

    # 生成所有可能的障礙物位置，排除起點 (0,0) 和終點 (rows-1,cols-1)
    all_positions = [(x, y) for x in range(rows) for y in range(cols)
                     if (x, y) not in [(0, 0), (rows - 1, cols - 1)]]

    # 隨機選擇障礙物位置
    obstacle_pos = random.sample(all_positions, num_obstacles)
    for x, y in obstacle_pos:
        maze[x][y] = 1  # 設定障礙物為 1

    maze[-1, -1] = 2  # 設定終點為 2
    return maze


# =============== 判斷迷宮是否可通行 (BFS) ===============
def is_valid_maze(maze):
    """ 使用 BFS 判斷迷宮是否可以通行 """
    rows, cols = maze.shape
    queue = deque([(0, 0)])
    visited = np.zeros_like(maze, dtype=bool)
    visited[0, 0] = True

    while queue:
        x, y = queue.popleft()
        if maze[x, y] == 2:
            return True  # 成功找到終點

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] != 1 and not visited[nx, ny]:
                visited[nx, ny] = True
                queue.append((nx, ny))

    return False  # 迷宮無法通行


# =============== 設定迷宮參數，生成迷宮並確定可執行 ===============
"""測試 100 * 100 * 0.45 , 步數大概 220 ~ 250 左右"""
while True:
    try:
        rows = int(input("請輸入迷宮行數 (越大計算越久): "))
        cols = int(input("請輸入迷宮列數 (越大計算越久): "))
        obstacle_ratio = float(input("請輸入障礙物比例 (建議不要超過0.45): "))
        max_steps = int(input("請輸入最多可走步數 (迷宮越大步數越多): "))
        train_times = int(input("請輸入訓練次數 (迷宮越大次數越大): "))
    except ValueError:
        print("請輸入正確的數字格式！")
        continue

    max_attempts = 100
    for _ in range(max_attempts):
        maze = ran_maze(rows, cols, obstacle_ratio)
        if is_valid_maze(maze):
            print("\n成功生成有效迷宮！")
            break
    else:
        print("\n 100 次嘗試後仍無法生成有效迷宮，請降低障礙比例！")
        continue

    break  # 迷宮生成成功，繼續下一步

# =============== Q-learning 參數 ===============
state_size = rows * cols
action_size = 4
Q = np.zeros((state_size, action_size))
alpha, gamma = 0.8, 0.9    # 學習率 跟 折扣因子


# =============== 位置及移動函數 ===============
def state_to_idx(x, y):
    """ 將 (x, y) 轉成 Q-table 的索引 """
    return x * cols + y


# 移動函數
def step(state, action, maze):
    """ AI 狀態更新 獎勵 跟有無到達終點 """
    x, y = state
    if action == 0: nx, ny = x - 1, y   # 上
    if action == 1: nx, ny = x + 1, y   # 下
    if action == 2: nx, ny = x, y - 1   # 左
    if action == 3: nx, ny = x, y + 1   # 右

    # 碰到邊界或障礙物給予大逞罰
    if nx < 0 or ny < 0 or nx >= rows or ny >= cols or maze[nx, ny] == 1:
        return state, -50, False
    # 到達終點
    elif maze[nx, ny] == 2:
        return (nx, ny), 100, True
    # 一般移動給小逞罰
    else:
        return (nx, ny), -1, False


# =============== 訓練 Q-learning ===============
for episode in range(train_times):
    state = (0, 0)
    done = False
    steps = 0

    # 動態調整探索率 (epsilon)
    epsilon = max(0.1, 0.9 - episode / train_times)

    while not done and steps < max_steps:
        state_idx = state_to_idx(*state)

        # epsilon-greedy 策略
        if np.random.random() < epsilon:
            action = np.random.choice(action_size)   # 隨機選擇動作
        else:
            action = np.argmax(Q[state_idx])         # 根據 Q-table 選擇最佳動作

        next_state, reward, done = step(state, action, maze)
        next_state_idx = state_to_idx(*next_state)

        # 更新 Q-table
        Q[state_idx, action] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action])

        state = next_state
        steps += 1


# =============== 找出 AI 走的最短路徑 ===============
state = (0, 0)
done = False
path = [state]
steps = 0


while not done and steps < max_steps:
    state_idx = state_to_idx(*state)
    action = np.argmax(Q[state_idx])
    next_state, reward, done = step(state, action, maze)

    if next_state == state:
        Q[state_idx, action] -= 10  # AI撞牆逞罰
        continue

    path.append(next_state)
    state = next_state
    steps += 1

if done:
    print(f"恭喜到達終點，走了 {steps} 步!!")
else:
    print(f"在回合內無法到達終點!")


# =============== 視覺化迷宮與最佳路徑 ===============
plt.figure(figsize=(cols / 2, rows / 2))
plt.imshow(maze, cmap='gray')

# 畫出 AI 的路徑
for i in range(len(path) - 1):
    plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], 'b--', linewidth=3)

# 標示起點與終點
plt.plot(path[0][1], path[0][0], "go", markersize=10, label="開始")
plt.plot(path[-1][1], path[-1][0], 'r*', markersize=10, label="終點")

plt.legend(loc='best')
plt.title("AI走最短路徑", fontsize=30)
plt.show()
