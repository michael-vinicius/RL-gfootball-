import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

# --- CONFIGURAÇÃO FASE 3 (FINAL) ---
raiz = "/gfootball/meu_projeto"
log_dir = os.path.join(raiz, "logs_fase3")
models_dir = os.path.join(raiz, "modelos_fase3")
best_model_dir = os.path.join(raiz, "melhor_modelo_fase3") # Onde ficará o CAMPEÃO MUNDIAL

os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)

# 1. AMBIENTE 5 vs 5 (O JOGO REAL)
# Stacked=True é obrigatório (cérebro já acostumou com memória)
env = football_env.create_environment(
    env_name='5_vs_5', 
    stacked=True,  
    representation='simple115',
    rewards='scoring,checkpoints', 
    render=False
)

# 2. AMBIENTE DE AVALIAÇÃO
env_eval = football_env.create_environment(
    env_name='5_vs_5', 
    stacked=True,  
    representation='simple115',
    rewards='scoring,checkpoints', 
    render=False
)

# 3. CARREGAR O CAMPEÃO DA FASE 2
# Tenta pegar o BEST model automático. Se não tiver, pega o FINAL.
caminho_best = os.path.join(raiz, "melhor_modelo_fase2", "best_model.zip")
caminho_final = os.path.join(raiz, "modelo_fase2_final.zip")

if os.path.exists(caminho_best):
    modelo_carregar = caminho_best
    print(f"Carregando o MELHOR momento da Fase 2: {modelo_carregar}")
elif os.path.exists(caminho_final):
    modelo_carregar = caminho_final
    print(f"Carregando o modelo FINAL da Fase 2: {modelo_carregar}")
else:
    print("ERRO: Não achei nenhum modelo da Fase 2. Verifique os nomes!")
    exit()

try:
    # Ajuste Fino para o Jogo Completo
    custom_objects = {
        "learning_rate": 0.0001, # Baixa para refinar
        "ent_coef": 0.01         # Padrão para manter estabilidade
    }
    model = PPO.load(modelo_carregar, env=env, custom_objects=custom_objects, print_system_info=True)
except Exception as e:
    print(f"Erro ao carregar: {e}")
    exit()

# --- CALLBACKS ---
# Salva histórico a cada 100k
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=models_dir, name_prefix='ppo_5v5_final')

# O Juiz avalia a cada 50k steps (no 5v5 jogos são demorados, melhor espaçar mais)
eval_callback = EvalCallback(
    env_eval, 
    best_model_save_path=best_model_dir,
    log_path=log_dir, 
    eval_freq=50000, 
    deterministic=True, 
    render=False
)

print("--- INICIANDO FASE 3: A COPA DO MUNDO (5vs5) ---")
print("O agente vai apanhar no começo até perceber que precisa variar as jogadas.")

# 3 Milhões de steps. (Pode parar antes se o ep_rew_mean ficar positivo)
model.learn(total_timesteps=3000000, callback=[checkpoint_callback, eval_callback])

model.save(os.path.join(raiz, "modelo_final_absoluto"))
print("Treinamento completo finalizado!")