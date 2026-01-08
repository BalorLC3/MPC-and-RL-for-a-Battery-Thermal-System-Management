from src.env_batt import BatteryCoolingEnv
from sbx import SAC
from utils.trainer import TrainExport
import os 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

env = BatteryCoolingEnv()

name = "sac_horizon"
model = SAC(
        env=env, 
        gamma=0.995,
        learning_rate=3e-4,
        policy= "MlpPolicy",
        buffer_size=100_000, # Large buffer for slow dynamics
        learning_starts=2_740, # Learning starts after one complete simulation of the system 
        ent_coef='auto_0.1',
        tau=0.005,
        verbose=1,
        seed=17
)

env = BatteryCoolingEnv()
trainer = TrainExport(
    model, 
    env, 
    path_prefix="results/" + name
)

trainer.train(total_timesteps=65_000)
