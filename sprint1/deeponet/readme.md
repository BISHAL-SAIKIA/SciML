# Physics-Informed DeepONet (PINO) Framework

This repository implements a **Physics-Informed Neural Operator (PINO)** using **Deep Operator Networks (DeepONet)** in PyTorch.

Unlike standard Physics-Informed Neural Networks (PINNs), this framework learns an **operator**:
> a mapping from functions to functions (e.g. forcing → solution)

This means a **single trained model generalizes across infinitely many PDE/ODE instances**, rather than solving one problem at a time.

---

## 🚀 What This Repository Demonstrates

- ✅ DeepONet architecture (Branch + Trunk networks)
- ✅ Physics-informed training using automatic differentiation
- ✅ Operator learning (function → function mapping)
- ✅ Clean, modular SciML codebase
- ✅ Research-grade Python packaging and imports

This codebase is designed to be **extensible** to:
- Burgers / Heat / Navier–Stokes equations
- Fourier Neural Operators (FNO)
- Hybrid data + physics training
- Multi-physics and probabilistic SciML

---

## 📁 Project Structure

It can be found in sprint1/deeponet/structures.txt


---

## 🧠 Design Philosophy

Each folder has **one clear responsibility**:

| Folder | Responsibility |
|------|---------------|
| `models/` | What is being learned (operators, architectures) |
| `physics/` | What physical laws must be satisfied |
| `data/` | What distribution of problems is sampled |
| `training/` | How learning is enforced |
| `utils/` | Shared utilities (normalization, seeds) |
| `experiments/` | Concrete runnable experiments |

This separation mirrors how **real SciML research codebases** (e.g. NVIDIA Modulus) are structured.

---

## ▶️ How to Run

From the directory **above** `deeponet/`:

```bash
python -m deeponet.experiments.run_training
```




## 🔄 Extending to Other Neural Operators and Physics

This framework is designed so that **physics**, **operator architecture**, and **training logic** are cleanly decoupled.  
This allows new neural operators and governing equations to be introduced with minimal code changes.

---

### 1️⃣ Adding Other Neural Operators (DeepONet → FNO → Graph Operators)

To introduce a new neural operator:

1. Implement the operator inside:<br>
models/
<br>
Examples: <br>
models/fno.py
models/graph_neural_operator.py
models/transformer_operator.py


Each operator should:
- Accept function inputs and coordinates (or grids)
- Return solution values in a consistent format
- Contain **only architecture-related code**

2. Select the desired operator in:
experiments/run_training.py


No changes are required in:
- `physics/`
- `training/`
- `data/`

This enables **fair comparison between different neural operators under the same physics constraints**.

---

### 2️⃣ Changing or Adding Physics (ODE → PDE → Multi-Physics)

To change the governing equation:

1. Add or modify residual definitions in:
physics/ <br><br>
Examples:
- physics/burgers.py
- physics/heat_equation.py
- physics/navier_stokes.py


Each physics module should:
- Define residuals using automatic differentiation
- Be independent of the chosen neural operator
- Return a residual tensor used for loss computation

2. Combine new residuals in:
```training/train_step.py```


The operator architecture remains unchanged.

---

### 3️⃣ Modifying the Architecture Independently of Physics

Because physics constraints are enforced externally:
- The same physics can be used with different operators
- The same operator can be trained with different physics

This separation allows experiments such as:
- DeepONet vs FNO on Burgers’ equation
- PINO vs data-driven operator learning
- Architecture ablations without physics changes

---

### 4️⃣ Guiding Principle

All extensions follow the same rule:

> **Change physics in `physics/`,  
> change operators in `models/`,  
> and select combinations in `experiments/`.**

As long as this separation is preserved, the framework remains scalable, interpretable, and research-ready.

---
