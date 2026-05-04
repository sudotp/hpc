import numpy as np
import time

# Network size
INPUT  = 784
HIDDEN = 256
OUTPUT = 10
BATCH  = 1000

# Activations
def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

# Forward pass for full batch
def forward(X, W1, b1, W2, b2):
    h = relu(X @ W1.T + b1)
    return sigmoid(h @ W2.T + b2)

if __name__ == "__main__":
    np.random.seed(42)

    # Weights and inputs
    W1 = np.random.randn(HIDDEN, INPUT)  * 0.1
    b1 = np.zeros(HIDDEN)
    W2 = np.random.randn(OUTPUT, HIDDEN) * 0.1
    b2 = np.zeros(OUTPUT)
    X  = np.random.rand(BATCH, INPUT)

    # Sequential
    t = time.time()
    for x in X:
        relu(x @ W1.T + b1)
    seq_time = time.time() - t

    # Parallel (batched NumPy)
    t = time.time()
    out = forward(X, W1, b1, W2, b2)
    par_time = time.time() - t

    print(f"Sequential Time : {seq_time:.4f} sec")
    print(f"Parallel Time   : {par_time:.4f} sec")
    print(f"Speedup         : {seq_time/par_time:.2f}x")
    print(f"Predicted Class : {np.argmax(out[0])}")

# Run: python3 P5_AI_ML.py