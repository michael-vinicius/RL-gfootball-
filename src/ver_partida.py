import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
import os

# === CONFIGURAÇÕES ===
# Caminho exato que apareceu no seu 'ls'
model_path = os.path.expanduser("~/gfootball_logs/FASE1_CORRIGIDO/models/ckpt_400000_steps")
# Caminho da normalização (provavelmente não existe ainda, mas o script vai tentar)
stats_path = os.path.expanduser("~/gfootball_logs/FASE1_CORRIGIDO/models/vec_normalize.pkl")
video_folder = "./prova_dos_gols"

os.makedirs(video_folder, exist_ok=True)

# === ADAPTER ===
class GfootballAdapter(gym.Env):
    def __init__(self, env):
        self.env = env
        obs_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_space.low, high=obs_space.high,
            shape=obs_space.shape, dtype='float32'
        )
        act_space = env.action_space
        self.action_space = gym.spaces.Discrete(act_space.n)
    def reset(self): return self.env.reset()
    def step(self, action): return self.env.step(action)

# === CRIA AMBIENTE QUE GRAVA ===
def make_env():
    env = football_env.create_environment(
        env_name='academy_empty_goal_close',
        stacked=True,
        representation='simple115',
        rewards='scoring',
        render=False,
        write_goal_dumps=True,     # Grava replay SE fizer gol
        write_full_episode_dumps=False,
        dump_frequency=1,          # Tenta gravar sempre
        logdir=video_folder
    )
    env = GfootballAdapter(env)
    return env

# === EXECUÇÃO ===
print(f"📂 Carregando modelo: {model_path}")
env = DummyVecEnv([make_env])

# Tenta carregar a normalização se ela existir
if os.path.exists(stats_path):
    print("✅ Arquivo de normalização encontrado! Carregando...")
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
else:
    print("⚠️ AVISO: vec_normalize.pkl não encontrado (normal pois o treino não acabou).")
    print("   O agente pode jogar de forma estranha/trêmula, mas deve marcar gol.")

# Carrega o modelo
model = PPO.load(model_path)

print("\n🎥 Iniciando teste de 50 tentativas...")
print(f"   Os vídeos dos gols serão salvos em: {video_folder}")
print("-" * 50)

obs = env.reset()
gols_confirmados = 0

try:
    for i in range(1000): # Roda até 1000 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Como é DummyVecEnv, reward é uma lista [reward_env1]
        if reward[0] > 0:
            gols_confirmados += 1
            print(f"⚽ GOL CONFIRMADO! ({gols_confirmados})")

        if done[0]:
            obs = env.reset()
            
except KeyboardInterrupt:
    pass

print("=" * 50)
print(f"📊 TOTAL DE GOLS PROVADOS: {gols_confirmados}")
print(f"📁 Verifique a pasta '{video_folder}' para baixar os vídeos (.dump)")