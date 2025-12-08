import gfootball.env as football_env
from stable_baselines3 import PPO
import gym  # Usamos 'gym' direto, pois instalamos a versão 0.21
import numpy as np
import os

# ==============================================================================
# 1. ADAPTADORES (Versão Gym 0.21 - Igual ao seu treino)
# ==============================================================================
class GfootballAdapter(gym.Env):
    def __init__(self, env):
        self.env = env
        old_obs = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=old_obs.low, high=old_obs.high, shape=old_obs.shape, dtype=old_obs.dtype
        )
        old_act = env.action_space
        if hasattr(old_act, 'n'):
            self.action_space = gym.spaces.Discrete(old_act.n)
        else:
            self.action_space = old_act
        self.metadata = env.metadata
        self.reward_range = env.reward_range

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

# Wrapper necessário para o modelo aceitar os inputs (mesmo que não use o reward aqui)
class ImprovedGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if obs is None: obs = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        return obs, reward, done, info

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
# Ajuste para onde seus modelos estão salvos
RAIZ_DIR = os.path.expanduser("~/gfootball_logs/meu_projeto")
MODEL_DIR = os.path.join(RAIZ_DIR, "checkpoints_improved_finishing_curriculum_v2")
VIDEO_DIR = os.path.join(RAIZ_DIR, "videos_partidas")

os.makedirs(VIDEO_DIR, exist_ok=True)

# Nome do arquivo do modelo que você quer ver
NOME_MODELO = "ppo_improved_200000_steps.zip" 
MODELO_PATH = os.path.join(MODEL_DIR, NOME_MODELO)

def assistir():
    if not os.path.exists(MODELO_PATH):
        print(f"❌ Erro: Modelo não encontrado em: {MODELO_PATH}")
        return

    print(f"📺 Carregando modelo: {NOME_MODELO}")
    
    try:
        # 1. Cria o ambiente com GRAVAÇÃO LIGADA e RENDER DESLIGADO
        env_legacy = football_env.create_environment(
            env_name='5_vs_5', 
            stacked=True,
            representation='simple115',
            render=False,                # <--- Importante: FALSE para servidor
            write_full_episode_dumps=True, # Salva o replay completo
            write_video=True,            # Salva o vídeo .avi
            logdir=VIDEO_DIR             # Onde salvar
        )
        
        # 2. Aplica os adaptadores
        env = GfootballAdapter(env_legacy)
        env = ImprovedGoalWrapper(env)
        
        # 3. Carrega o cérebro treinado
        model = PPO.load(MODELO_PATH, env=env)
        print("✅ Modelo carregado! Gravando partida...")

    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        return

    # Joga uma partida
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    while not done:
        # deterministic=True usa a melhor jogada (sem aleatoriedade)
        # deterministic=False deixa ele ser criativo (bom para ver dribles)
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"   ... gravando frame {steps}")

    print("-" * 30)
    print(f"📊 FIM DE JOGO!")
    print(f"   Passos: {steps}")
    print(f"   Reward Acumulado: {total_reward:.2f}")
    print(f"   Vídeo salvo em: {VIDEO_DIR}")
    print("-" * 30)
    
    env.close()

if __name__ == "__main__":
    assistir()