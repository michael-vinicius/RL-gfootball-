import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

# --- CONFIGURAÇÃO FASE 2 ---
raiz = "/gfootball/meu_projeto"
log_dir = os.path.join(raiz, "logs_fase2")
models_dir = os.path.join(raiz, "modelos_fase2")
best_model_dir = os.path.join(raiz, "melhor_modelo_fase2") # Pasta para o campeão

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

# 1. AMBIENTE DE TREINO (Onde ele sua a camisa)
env = football_env.create_environment(
    env_name='academy_3_vs_1_with_keeper', 
    stacked=True,  
    representation='simple115',
    rewards='scoring,checkpoints', 
    render=False
)

# 2. AMBIENTE DE AVALIAÇÃO (O JUIZ) - NOVO!
# É necessário ter um ambiente separado para testar sem interferir no treino
env_eval = football_env.create_environment(
    env_name='academy_3_vs_1_with_keeper', 
    stacked=True,  
    representation='simple115',
    rewards='scoring,checkpoints', 
    render=False
)

# 3. CARREGAR O CAMPEÃO DA FASE 1
modelo_fase1 = os.path.join(raiz, "modelo_fase1_campeao.zip")
print(f"Carregando o Artilheiro da Fase 1: {modelo_fase1}")

try:
    custom_objects = {
        "learning_rate": 0.0001,
        "ent_coef": 0.03
    }
    model = PPO.load(modelo_fase1, env=env, custom_objects=custom_objects, print_system_info=True)
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")
    exit()

# --- CALLBACKS (Os Ajudantes) ---

# 1. O Historiador (Salva tudo a cada 50k, por garantia)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=models_dir, name_prefix='fase2_passe')

# 2. O Juiz (Testa a cada 10k e salva o MELHOR) - NOVO!
eval_callback = EvalCallback(
    env_eval, 
    best_model_save_path=best_model_dir,
    log_path=log_dir, 
    eval_freq=10000, # A cada 10 mil passos ele para e testa
    deterministic=True, 
    render=False
)

print("--- INICIANDO FASE 2: APRENDER A TOCAR A BOLA ---")
print("Agora com salvamento automático do 'best_model.zip'")

# Passamos uma LISTA de callbacks
model.learn(total_timesteps=800000, callback=[checkpoint_callback, eval_callback])

model.save(os.path.join(raiz, "modelo_fase2_final"))
print("Fase 2 finalizada!")