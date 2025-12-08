import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
import os

# === CONFIGURAÇÕES ===
# Onde está o modelo da Fase 1 (o "cérebro" inteligente)
# IMPORTANTE: Confirme se o nome do arquivo zip está correto na sua pasta
modelo_fase1 = os.path.expanduser("~/gfootball_logs/FASE1_CORRIGIDO/models/FASE1_FINAL") 
# Se o FASE1_FINAL não existir, use o último checkpoint: ckpt_400000_steps

# Onde vamos salvar a Fase 2
log_dir = os.path.expanduser("~/gfootball_logs/FASE2_GOLEIRO")
models_dir = f"{log_dir}/models"
os.makedirs(models_dir, exist_ok=True)

print("\n" + "=" * 80)
print("🚀 FASE 2 - AGORA TEM GOLEIRO! 🚀")
print("   Carregando conhecimentos da Fase 1...")
print("=" * 80 + "\n")

# === ADAPTER (Igual à Fase 1) ===
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

# === CONTADOR DE GOLS ===
class GoalCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.goals = 0
    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [])
        if rewards is not None: self.goals += sum(1 for r in rewards if r > 0)
        return True
    def _on_rollout_end(self) -> None:
        self.logger.record('rollout/goals', self.goals)
        self.goals = 0

# === CRIAÇÃO DO AMBIENTE ===
def make_env():
    env = football_env.create_environment(
        env_name='academy_run_to_score_with_keeper', 
        stacked=True,
        representation='simple115',
        # MUDANÇA AQUI: Adicionamos 'checkpoints'
        rewards='scoring,checkpoints', 
        render=False
    )
    env = GfootballAdapter(env)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # 1. Cria 8 ambientes paralelos
    vec_env = DummyVecEnv([make_env for _ in range(8)])
    
    # 2. Cria nova normalização
    # Não carregamos a normalização antiga (pkl) porque o campo/distâncias mudaram.
    # O agente vai re-aprender a "escala" do mundo rapidamente.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 3. Carrega o CÉREBRO da Fase 1
    # custom_objects serve para ajustar caso a versão do python/biblioteca mude, mas geralmente não precisa
    print(f"🧠 Carregando pesos de: {modelo_fase1}")
    
    # Carregamos o modelo, mas passamos o 'env' novo para ele se conectar ao novo desafio
    model = PPO.load(modelo_fase1, env=vec_env, device="auto")
    
    # Ajustamos parâmetros para refino (opcional, mas ajuda)
    model.learning_rate = 0.0002       # Taxa menor para não esquecer o que já sabe
    model.ent_coef = 0.03              # Um pouco de exploração para tentar driblar
    model.n_steps = 2048
    model.tensorboard_log = log_dir

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=models_dir, name_prefix='ckpt_fase2'),
        GoalCounter(),
    ]

    print("\n🥊 TREINANDO CONTRA O GOLEIRO (1 Milhão de steps)...")
    # Treina por mais tempo pois agora é mais difícil
    model.learn(total_timesteps=1_000_000, callback=callbacks, progress_bar=True)

    model.save(f"{models_dir}/FASE2_FINAL")
    vec_env.save(f"{models_dir}/vec_normalize_fase2.pkl")
    vec_env.close()

    print("✅ FASE 2 COMPLETA!")