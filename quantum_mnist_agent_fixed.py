# Install required packages if needed:
# pip install pennylane pennylane-qiskit torch numpy

import pennylane as qml
from pennylane import numpy as np
import torch
import random

# -------------------------------
# Tic-Tac-Toe Environment
# -------------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [0]*9  # 0 = empty, 1 = AI, -1 = opponent
        self.done = False
        return self.get_state()

    def get_state(self):
        return np.array(self.board, dtype=float)

    def step(self, action, player):
        if self.board[action] != 0:
            return self.get_state(), -1, True  # invalid move
        self.board[action] = player
        self.done = self.check_win(player) or all(cell != 0 for cell in self.board)
        reward = 1 if self.check_win(player) else 0
        return self.get_state(), reward, self.done

    def check_win(self, player):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        return any(all(self.board[i]==player for i in combo) for combo in wins)

    def render(self):
        symbols = {1:'X', -1:'O', 0:' '}
        for i in range(3):
            print(" | ".join([symbols[self.board[3*i+j]] for j in range(3)]))
            if i < 2:
                print("--+---+--")
        print("\n")

# -------------------------------
# Quantum Circuit & Agent
# -------------------------------
n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

# Fix: input argument name must be 'inputs' for TorchLayer
def quantum_circuit(inputs, params):
    for i in range(n_qubits):
        qml.RY(inputs[i] * np.pi, wires=i)
        qml.RZ(params[i], wires=i)
    qml.CNOT(wires=[0,1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"params": n_qubits}

# Create TorchLayer
qlayer = qml.qnn.TorchLayer(qml.QNode(quantum_circuit, dev), weight_shapes)

class QuantumTicTacToeAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qlayer
        self.fc = torch.nn.Linear(n_qubits, 9)  # 9 possible moves

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
            # Mask invalid moves
            mask = torch.tensor([0 if state[i]!=0 else 1 for i in range(9)], dtype=torch.float32)
            action_probs = action_probs * mask
            action_probs = action_probs / torch.sum(action_probs)
            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done = env.step(action, player=1)

            # Random opponent move
            if not done:
                empty_cells = [i for i, cell in enumerate(env.board) if cell==0]
                if empty_cells:
                    opp_action = random.choice(empty_cells)
                    _, _, done = env.step(opp_action, player=-1)

            # Policy gradient update
            loss = -torch.log(action_probs[action]+1e-10) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if (episode+1) % 50 == 0:
            print(f"Episode {episode+1} completed")

# -------------------------------
# Play Game vs Human
# -------------------------------
def play_game(agent, env):
    state = env.reset()
    done = False
    print("Game Start!")
    env.render()
    while not done:
        # AI move
        state_tensor = torch.tensor(state[:n_qubits], dtype=torch.float32)
        action_probs = agent(state_tensor)
        mask = torch.tensor([0 if state[i]!=0 else 1 for i in range(9)], dtype=torch.float32)
        action_probs = action_probs * mask
        action_probs = action_probs / torch.sum(action_probs)
        action = torch.multinomial(action_probs, 1).item()
        state, _, done = env.step(action, player=1)
        print("AI plays:")
        env.render()
        if done:
            print("Game Over!")
            break

        # Human move
        empty_cells = [i for i, cell in enumerate(state) if cell==0]
        human_move = -1
        while human_move not in empty_cells:
            try:
                human_move = int(input(f"Your move (choose empty cell 0-8): "))
            except:
                continue
        state, _, done = env.step(human_move, player=-1)
        print("You play:")
        env.render()
        if done:
            print("Game Over!")
            break

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    env = TicTacToeEnv()
    agent = QuantumTicTacToeAgent()
    print("Training Quantum Tic-Tac-Toe Agent...")
    train_agent(agent, env, episodes=500)
    print("Training completed!\n")
    play_game(agent, env)
