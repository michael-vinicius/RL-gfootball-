import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym
import os

# Adaptador simples apenas para converter formatos, sem mexer na recompensa
class GfootballAdapter(gym.Env):
    def __init__(self, env):
        self.env = env
        obs_space = env.observation_space
        # Define explicitamente o espaço como float32 para evitar erros de tipo
        self.observation_space = gym.spaces.Box(
            low=obs_space.low, high=obs_space.high,
            shape=obs_space.shape, dtype='float32'
        )
        act_space = env.action_space
        self.action_space = gym.spaces.Discrete(act_space.n)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

# Callback simples para logar gols no console
class GoalCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.goals = 0

    def _on_step(self) -> bool:
        # Verifica se houve recompensa positiva (gol) nos infos
        # O gfootball com rewards='scoring' retorna reward=1 no gol
        rewards = self.locals.get('rewards', [])
        if rewards is not None:
             # Soma quantos ambientes tiveram reward > 0 (gol) neste step
             self.goals += sum(1 for r in rewards if r > 0)
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('rollout/goals', self.goals)
        print(f"📊 Gols neste Rollout: {self.goals}")
        self.goals = 0

if __name__ == "__main__":
    log_dir = os.path.expanduser("~/gfootball_logs/FASE1_CORRIGIDO")
    models_dir = f"{log_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 FASE 1 - TREINAMENTO DE CHUTE (CORRIGIDO) 🚀")
    print("=" * 80 + "\n")

    def make_env():
        # rewards='scoring' já garante +1 no gol e 0 no resto
        env = football_env.create_environment(
            env_name='academy_empty_goal_close',
            stacked=True,
            representation='simple115',
            rewards='scoring', 
            render=False
        )
        env = GfootballAdapter(env)
        env = Monitor(env) # Monitora recompensas puras
        return env

    # Criação dos ambientes
    vec_env = DummyVecEnv([make_env for _ in range(8)])
    
    # IMPORTANTE: Normalização das observações e recompensas
    # Isso ajuda a rede neural a entender os dados do simple115
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    print("🤖 Inicializando PPO com alta exploração...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=0.0003,
        n_steps=2048,           # Aumentei steps para coletar mais dados por update
        batch_size=256,         # Batch maior para estabilidade
        n_epochs=4,
        gamma=0.993,            # Gamma levemente maior
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,          # Entropia maior (2%) para forçar exploração inicial
        verbose=1,
        tensorboard_log=log_dir,
        device="auto",
    )

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path=models_dir, name_prefix='ckpt'),
        GoalCounter(),
    ]

    print("\n🎯 TREINANDO...")
    # 500k steps devem ser suficientes para esse cenário simples se estiver funcionando
    model.learn(total_timesteps=500_000, callback=callbacks, progress_bar=True)

    # Salva o modelo e também as estatísticas de normalização
    model.save(f"{models_dir}/FASE1_FINAL")
    vec_env.save(f"{models_dir}/vec_normalize.pkl") # Salvar status da normalização é crucial
    
    vec_env.close()
    print("✅ Treino finalizado.")