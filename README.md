# Kármán Vortex Street Energy Harvesting

PPO agent trained to reduce UAV thrust usage by exploiting a Kármán vortex
street, via curriculum learning over three difficulty stages. AE4350
Bio-Inspired Intelligence and Learning for Aerospace Applications, TU Delft.

The agent learns to ride the wake rather than fly through it — 9.0x less
thrust and 4.3x lower relative airspeed than a random policy, with goal
completion improving from 10% to 100%.

## Structure

- `env/` — vortex flow model, UAV dynamics, Gymnasium environment
- `train/` — curriculum config and training entry point
- `analysis/` — figure generation, sensitivity sweep, generalization sweep
  

## Method

A regularized vortex-blob flow model (Strouhal number 0.2, viscous core
decay) is coupled to a point-mass UAV with a parabolic drag polar. A PPO
agent (Stable-Baselines3) is trained through three curriculum stages of
increasing environmental difficulty — regular street, moderate noise, then
full noise + crosswind + randomized starting position.

## Run

```
pip install -r requirements.txt
python train/train.py
python analysis/visualize_agent.py
python analysis/sensitivity.py
python analysis/generalization_check.py
```


