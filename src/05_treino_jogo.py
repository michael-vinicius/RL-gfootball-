import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
import os

# === CONFIGURAÇÕES ===
# Carregamos o modelo "Coletivo" da Fase 4
# AJUSTE ESTE CAMINHO conforme seus logs anteriores
modelo_anterior = os.path.expanduser("~/gfootball_logs/FASE4_PASSE/models/FASE4_FINAL") 

log_dir = os.path.expanduser("~/gfootball_logs/FASE5_FINAL")
models_dir = f"{log_dir}/models"
os.makedirs(models_dir, exist_ok=True)

print("\n" + "=" * 80)
print("🏆 FASE 5 - A GRANDE FINAL (5 vs 5) 🏆")
print("   Objetivo: Jogar futebol completo.")
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
        # O CENÁRIO REAL: Futebol 5 contra 5
        env_name='5_vs_5', 
        stacked=True,
        representation='simple115',
        # Scoring puro é muito difícil no 5v5 logo de cara.
        # Mantemos checkpoints para ele não ficar perdido no meio campo.
        rewards='scoring,checkpoints', 
        render=False
    )
    env = GfootballAdapter(env)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # Vamos aumentar para 16 ambientes em paralelo se o servidor aguentar
    # Se der erro de memória, volte para 8.
    n_envs = 8 
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print(f"🧠 Carregando o Jogador Completo da Fase 4: {modelo_anterior}")
    
    try:
        model = PPO.load(modelo_anterior, env=vec_env, device="auto")
        print("✅ Modelo carregado! O jogo vai começar...")
    except Exception as e:
        print(f"❌ ERRO: Não achei o modelo da Fase 4: {e}")
        exit()
    
    # === AJUSTES FINAIS ===
    # Mantemos learning rate baixo para refinar a estratégia
    model.learning_rate = 0.00005 
    model.ent_coef = 0.02 # Levemente menor, queremos menos "loucura" e mais consistência
    model.n_steps = 2048
    model.tensorboard_log = log_dir

    callbacks = [
        # Salvamos com menos frequência pois o treino é longo
        CheckpointCallback(save_freq=100_000, save_path=models_dir, name_prefix='ckpt_5v5'),
        GoalCounter(),
    ]

    print("\n🏆 TREINANDO A PARTIDA FINAL (3 Milhões de steps)...")
    # Isso vai demorar algumas horas, mas é o teste final
    model.learn(total_timesteps=3_000_000, callback=callbacks, progress_bar=True)

    model.save(f"{models_dir}/CAMPEAO_5V5")
    vec_env.save(f"{models_dir}/vec_normalize_5v5.pkl")
    vec_env.close()

    print("✅ PROJETO CONCLUÍDO! VOCÊ CRIOU UMA IA JOGADORA DE FUTEBOL!")