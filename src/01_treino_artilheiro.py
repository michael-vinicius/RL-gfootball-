import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch.nn as nn

# --- CONFIGURAÇÃO TURBO ---
raiz = "/gfootball/meu_projeto"
log_dir = os.path.join(raiz, "logs_turbo")
models_dir = os.path.join(raiz, "modelos_turbo")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# 1. AMBIENTE COM MEMÓRIA (STACKED) E INCENTIVOS
# stacked=True: Empilha 4 quadros. Ele percebe movimento e velocidade do inimigo.
# rewards='scoring,checkpoints': Checkpoints fazem ele andar para frente!
env = football_env.create_environment(
    env_name='academy_run_to_score_with_keeper', 
    stacked=True,
    representation='simple115',
    rewards='scoring,checkpoints', 
    render=False
)

# 2. ARQUITETURA DA REDE NEURAL MAIS PROFUNDA
# net_arch: Cria um "cérebro" maior para entender táticas de drible.
# pi: [256, 256] -> Camadas para decidir a ação (chutar, correr)
# vf: [256, 256] -> Camadas para avaliar se a jogada é boa
policy_kwargs = dict(
    activation_fn=nn.ReLU,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

print("--- INICIANDO TREINO TURBO (Checkpoints + Memória) ---")
print("O agente vai aprender a correr para o gol imediatamente.")

model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.02, # Entropia média para ele tentar dribles diferentes
    policy_kwargs=policy_kwargs # Aplica o cérebro maior
)

# Salva a cada 50k
callback = CheckpointCallback(save_freq=50000, save_path=models_dir, name_prefix='turbo_chute')

# 500k a 1 Milhão deve ser suficiente para ele virar profissional nesse cenário
model.learn(total_timesteps=800000, callback=callback)

model.save(os.path.join(raiz, "modelo_turbo_final"))
print("Treino Turbo finalizado!")