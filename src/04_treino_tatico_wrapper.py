import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gym
import numpy as np
import os

# --- WRAPPER TÁTICO INTELIGENTE ---
class TacticalRefinedWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TacticalRefinedWrapper, self).__init__(env)
        self.last_ball_owned_team = -1
        self.last_ball_x = 0
        self.steps_with_ball_stopped = 0
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # --- LEITURA DO CAMPO ---
        # 88: Ball X (-1 a 1) | 89: Ball Y (-0.42 a 0.42)
        # 97: Posse (0 = Nós, 1 = Eles, -1 = Ninguém)
        ball_x, ball_y = obs[88], obs[89]
        ball_vel_x, ball_vel_y = obs[94], obs[95]
        ball_owned_team = obs[97]
        
        # 1. ECONOMIA DO GOL (Prioridade Absoluta)
        if reward == 1.0: 
            reward = 20.0 # Aumentei para 20. O gol é a única coisa que importa.
            print("!!! GOLAÇO !!!")
            
        # 2. PUNIÇÃO SEVERA POR TOMAR GOL
        # Com dificuldade 0.85, isso vai acontecer. Ele precisa odiar isso.
        if reward == -1.0:
            reward = -10.0 
            
        # 3. CORREÇÃO DO PASSE (Fim do "Toque Inútil")
        # Se ele der passe (9, 10, 11), analisamos a qualidade:
        if action in [9, 10, 11]:
            # Regra A: Passe no campo de defesa (X < 0) NÃO dá prêmio.
            # Regra B: Passe só dá prêmio se a bola estiver indo para o ataque.
            if ball_x > -0.1: 
                reward += 0.05 # Pequeno incentivo
            else:
                reward += 0.0 # Zero incentivo para toquinho na defesa.
        
        # 4. PROGRESSÃO REAL (O que define um bom ataque)
        # Se a bola avançou em direção ao gol (X aumentou)
        distancia_avancada = ball_x - self.last_ball_x
        if distancia_avancada > 0.01:
            # Recompensa alta por ganhar terreno
            reward += 3.0 * distancia_avancada 

        # 5. ANTI-CERA (Punição por segurar a bola)
        if ball_owned_team == 0:
            speed = np.sqrt(ball_vel_x**2 + ball_vel_y**2)
            # Se a bola está muito lenta e é nossa, pune.
            if speed < 0.02:
                self.steps_with_ball_stopped += 1
                if self.steps_with_ball_stopped > 10:
                    reward -= 0.01 # Punição dolorida
            else:
                self.steps_with_ball_stopped = 0
        
        # 6. CÃO DE GUARDA (Roubada de Bola)
        # Se a bola era deles (1) e virou nossa (0) -> BÔNUS
        if self.last_ball_owned_team == 1 and ball_owned_team == 0:
            reward += 2.0 # Aumentei para 2.0. Roubar a bola vale muito!
            print("!!! ROUBADA DE BOLA !!!")

        # 7. LIMITES DO CAMPO
        # Se chegar muito perto da lateral sem necessidade, pune levemente
        if abs(ball_y) > 0.41:
            reward -= 0.05

        self.last_ball_owned_team = ball_owned_team
        self.last_ball_x = ball_x
        
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_ball_owned_team = -1
        self.last_ball_x = obs[88]
        self.steps_with_ball_stopped = 0
        return obs

# --- CONFIGURAÇÃO ---
raiz = "/gfootball/meu_projeto"
log_dir = os.path.join(raiz, "logs_tatico_final")
models_dir = os.path.join(raiz, "modelos_tatico_final")
best_model_dir = os.path.join(raiz, "melhor_modelo_tatico_final")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

# DIFICULDADE ALTA (0.85)
# O adversário vai jogar sério. Vai fazer gol se deixarem.
config_hard = {'difficulty': 0.85}

# 1. AMBIENTE
env = football_env.create_environment(
    env_name='5_vs_5', 
    stacked=True,  
    representation='simple115',
    rewards='scoring,checkpoints', 
    other_config_options=config_hard,
    render=False
)
env = TacticalRefinedWrapper(env) # Aplica as novas regras

# 2. AVALIAÇÃO
env_eval = football_env.create_environment(
    env_name='5_vs_5', 
    stacked=True,  
    representation='simple115',
    rewards='scoring,checkpoints', 
    other_config_options=config_hard,
    render=False
)
env_eval = TacticalRefinedWrapper(env_eval)

# 3. CARREGAR (Obrigatório usar o backup 1.58, que é o único são)
caminho_base = os.path.join(raiz, "campeoes_eternos", "modelo_1.58_fase3.zip")

print(f"Carregando a base sólida (1.58) para o desafio final: {caminho_base}")

try:
    custom_objects = {
        "learning_rate": 0.0001, 
        "ent_coef": 0.02 
    }
    model = PPO.load(
        caminho_base, 
        env=env, 
        custom_objects=custom_objects, 
        tensorboard_log=log_dir, # LOGS ATIVADOS!
        print_system_info=True
    )
except Exception as e:
    print(f"Erro crítico: {e}")
    exit()

# --- CALLBACKS ---
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=models_dir, name_prefix='ppo_tatico_final')
eval_callback = EvalCallback(env_eval, best_model_save_path=best_model_dir, log_path=log_dir, eval_freq=50000, deterministic=True, render=False)

print("--- TREINO TÁTICO FINAL ---")
print("Dificuldade: 0.85 (Adversário Agressivo)")
print("Gol = +20 | Roubada = +2 | Passe Inútil = 0")

model.learn(total_timesteps=3000000, callback=[checkpoint_callback, eval_callback])
model.save(os.path.join(raiz, "modelo_final_absoluto_v2"))