import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
import os

# === CONFIGURAÇÕES ===
# Ajuste o caminho para onde está o FASE2_FINAL no seu SERVIDOR
modelo_anterior = os.path.expanduser("~/RL-gfootball-/src/modelos_fase2/FASE2_FINAL") 
# Se der erro de arquivo não encontrado, use o checkpoint mais recente da fase 2

log_dir = os.path.expanduser("~/gfootball_logs/FASE3_DRIBLE")
models_dir = f"{log_dir}/models"
os.makedirs(models_dir, exist_ok=True)

print("\n" + "=" * 80)
print("🚀 FASE 3 - O DUELO (3 vs 1) 🚀")
print("   Objetivo: Enfrentar zagueiro + goleiro (Drible ou Passe)")
print("=" * 80 + "\n")

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

# === CONTADOR DE GOLS ===
class GoalCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.goals = 0
    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [])
        if rewards is not None: 
            self.goals += sum(1 for r in rewards if r > 0.8)
        return True
    def _on_rollout_end(self) -> None:
        self.logger.record('rollout/goals', self.goals)
        self.goals = 0

# === AMBIENTE ===
def make_env():
    env = football_env.create_environment(
        # CORREÇÃO: Usando cenário oficial que tem zagueiro
        env_name='academy_3_vs_1_with_keeper', 
        stacked=True,
        representation='simple115',
        rewards='scoring,checkpoints', 
        render=False
    )
    env = GfootballAdapter(env)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    vec_env = DummyVecEnv([make_env for _ in range(8)])
    
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print(f"🧠 Carregando o Craque da Fase 2: {modelo_anterior}")
    
    # Carrega o modelo anterior
    # Tenta carregar. Se falhar, verifique o caminho do arquivo zip
    try:
        model = PPO.load(modelo_anterior, env=vec_env, device="auto")
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ ERRO ao carregar modelo Fase 2: {e}")
        print("   Verifique se o caminho em 'modelo_anterior' está apontando para o arquivo .zip correto.")
        exit()
    
    # === AJUSTES FINOS PARA FASE 3 ===
    model.learning_rate = 0.0001 
    model.ent_coef = 0.03        
    model.n_steps = 2048
    model.tensorboard_log = log_dir

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=models_dir, name_prefix='ckpt_fase3'),
        GoalCounter(),
    ]

    print("\n🤼 TREINANDO (1.5 Milhões de steps)...")
    model.learn(total_timesteps=1_500_000, callback=callbacks, progress_bar=True)

    model.save(f"{models_dir}/FASE3_FINAL")
    vec_env.save(f"{models_dir}/vec_normalize_fase3.pkl")
    vec_env.close()

    print("✅ FASE 3 COMPLETA!")