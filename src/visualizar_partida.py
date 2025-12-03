import gfootball.env as football_env
from stable_baselines3 import PPO
import numpy as np
import os

# --- CONFIGURAÇÃO ---
raiz = "/gfootball/meu_projeto"

# Tenta pegar o best_model (Campeão). Se não tiver, pega o Final (Último salvo).
caminho_best = os.path.join(raiz, "melhor_modelo_fase5", "best_model.zip")
caminho_final = os.path.join(raiz, "modelo_final_hardcore.zip") # Ou o nome que você salvou no final

if os.path.exists(caminho_best):
    print(f"✅ Encontrei o 'best_model.zip'! Usando ele.")
    modelo_para_testar = caminho_best
elif os.path.exists(caminho_final):
    print(f"⚠️ Não achei o best_model. Usando o modelo final do treino.")
    modelo_para_testar = caminho_final
else:
    print("❌ Pânico: Não achei nenhum modelo dessa fase (nem best, nem final).")
    print("Verifique os nomes na pasta /gfootball/meu_projeto/")
    exit()

# Configuração igual ao treino (0.25 difficulty)
config_gradual = {'difficulty': 0.25}

# Cria ambiente
try:
    # Tenta com render se tiver monitor, senão sem render
    env = football_env.create_environment(
        env_name='5_vs_5', 
        stacked=True,  
        representation='simple115',
        rewards='scoring', # Só queremos saber de GOL agora
        other_config_options=config_gradual,
        render=False 
    )
except:
    print("Erro ao criar ambiente.")
    exit()

print(f"Carregando cérebro: {modelo_para_testar}")
model = PPO.load(modelo_para_testar, env=env)

print("\n--- INICIANDO PROVA FINAL (10 JOGOS) ---")
print("Dificuldade: 0.25 (Gradual)")

resultados = []

for i in range(1, 11):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        score += reward
    
    resultados.append(score)
    
    # Placar visual do jogo
    status = "VITÓRIA" if score > 0 else "EMPATE/DERROTA"
    print(f"Jogo {i}: Placar {score:.1f} -> {status}")

media = np.mean(resultados)
print("-" * 30)
print(f"📊 MÉDIA FINAL: {media:.4f}")
print("-" * 30)

if media > 1.58:
    print("🚀 SUCESSO! O modelo evoluiu (Melhor que 1.58).")
    print("Próximo passo: Aumentar dificuldade para 0.60.")
elif media > 0.5:
    print("✅ BOM. O modelo está ganhando, mas não superou drasticamente o anterior.")
else:
    print("⚠️ ALERTA. O modelo piorou ou estagnou.")