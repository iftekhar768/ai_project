# Install required packages if needed:
# pip install pennylane pennylane-qiskit torch numpy pygame

import pennylane as qml
from pennylane import numpy as np
import torch
import random
import pygame
import sys

# -------------------------------
# Tic-Tac-Toe Environment
# -------------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0]*9  # 0 = empty, 1 = AI, -1 = human
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array(self.board, dtype=float)

    def step(self, action, player):
        if self.board[action] != 0:
            return self.get_state(), -1, True
        self.board[action] = player
        self.done = self.check_win(player) or all(cell != 0 for cell in self.board)
        reward = 1 if self.check_win(player) else 0
        return self.get_state(), reward, self.done

    def check_win(self, player):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        return any(all(self.board[i]==player for i in combo) for combo in wins)

# -------------------------------
# Quantum Circuit & Agent
# -------------------------------
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

def quantum_circuit(inputs, params):
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
        qml.RZ(params[i], wires=i)
    qml.CNOT(wires=[0,1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"params": n_qubits}
qlayer = qml.qnn.TorchLayer(qml.QNode(quantum_circuit, dev), weight_shapes)

class QuantumTicTacToeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qlayer
        self.fc = torch.nn.Linear(n_qubits, 9)

    def forward(self, x):
        x = self.q_layer(x)
        return torch.softmax(self.fc(x), dim=-1)

# -------------------------------
# Training Loop
# -------------------------------
def train_agent(agent, env, episodes=500):
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.05)
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state[:n_qubits], dtype=torch.float32)
            action_probs = agent(state_tensor)
            mask = torch.tensor([0 if state[i]!=0 else 1 for i in range(9)], dtype=torch.float32)
            action_probs = action_probs * mask
            action_probs = action_probs / torch.sum(action_probs)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done = env.step(action, player=1)

            if not done:
                empty_cells = [i for i, cell in enumerate(env.board) if cell==0]
                if empty_cells:
                    opp_action = random.choice(empty_cells)
                    _, _, done = env.step(opp_action, player=-1)

            loss = -torch.log(action_probs[action]+1e-10) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1} completed")

# -------------------------------
# Pygame Interface
# -------------------------------
CELL_SIZE = 100
LINE_WIDTH = 5
BOARD_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
X_COLOR = (84, 84, 84)
O_COLOR = (242, 235, 211)
SCREEN_SIZE = CELL_SIZE*3

def draw_board(screen, board):
    screen.fill(BOARD_COLOR)
    # Draw grid lines
    pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE,0),(CELL_SIZE,SCREEN_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (CELL_SIZE*2,0),(CELL_SIZE*2,SCREEN_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0,CELL_SIZE),(SCREEN_SIZE,CELL_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0,CELL_SIZE*2),(SCREEN_SIZE,CELL_SIZE*2), LINE_WIDTH)
    
    # Draw X and O
    for i in range(9):
        x = (i % 3) * CELL_SIZE
        y = (i // 3) * CELL_SIZE
        if board[i] == 1:
            pygame.draw.line(screen, X_COLOR, (x+15, y+15), (x+CELL_SIZE-15, y+CELL_SIZE-15), LINE_WIDTH)
            pygame.draw.line(screen, X_COLOR, (x+CELL_SIZE-15, y+15), (x+15, y+CELL_SIZE-15), LINE_WIDTH)
        elif board[i] == -1:
            pygame.draw.circle(screen, O_COLOR, (x+CELL_SIZE//2, y+CELL_SIZE//2), CELL_SIZE//2-15, LINE_WIDTH)

def check_click(pos):
    x, y = pos
    col = x // CELL_SIZE
    row = y // CELL_SIZE
    return row*3 + col

def play_game_pygame(agent, env):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Quantum Tic-Tac-Toe")
    clock = pygame.time.Clock()
    running = True
    state = env.reset()
    draw_board(screen, state)
    pygame.display.update()
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN and not done:
                human_move = check_click(pygame.mouse.get_pos())
                if state[human_move]==0:
                    state, _, done = env.step(human_move, -1)

                    # AI move
                    if not done:
                        state_tensor = torch.tensor(state[:n_qubits], dtype=torch.float32)
                        action_probs = agent(state_tensor)
                        mask = torch.tensor([0 if state[i]!=0 else 1 for i in range(9)], dtype=torch.float32)
                        action_probs = action_probs * mask
                        action_probs = action_probs / torch.sum(action_probs)
                        action = torch.multinomial(action_probs, 1).item()
                        state, _, done = env.step(action, 1)

        draw_board(screen, state)
        pygame.display.update()
        clock.tick(30)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    env = TicTacToeEnv()
    agent = QuantumTicTacToeAgent()
    print("Training Quantum Tic-Tac-Toe Agent...")
    train_agent(agent, env, episodes=500)
    print("Training completed!")
    play_game_pygame(agent, env)
