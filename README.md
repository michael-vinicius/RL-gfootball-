# ⚽ Google Football RL Agent (PPO + Curriculum Learning)

Este projeto treina agentes autônomos para jogar futebol (5vs5) no ambiente **Google Research Football** utilizando Aprendizado por Reforço (Stable-Baselines3 PPO).

## �� Evolução e Estratégia (Curriculum Learning)

O treinamento foi dividido em fases para facilitar o aprendizado progressivo:

1.  **Fase 1 (Artilheiro):** Treino focado apenas em chutar ao gol (Cenário: Academy). *Status: Concluído.*
2.  **Fase 2 (Coletivo):** Treino 3vs1 para aprender a passar a bola. *Status: Concluído.*
3.  **Fase 3 (Competitivo):** Jogo completo 5vs5 contra bot Easy. **Resultado Alcançado: Score médio 1.58.**
4.  **Fase 4 (Tático - Em Andamento):** Refinamento com `Custom Wrappers` para corrigir vícios de comportamento (ex: segurar a bola na defesa), punir a passividade e incentivar a marcação pressão.

## 📂 Estrutura do Repositório

- `src/`: Scripts de treinamento numerados por fase.
- `models/`: Checkpoints dos modelos treinados (incluindo o campeão da Fase 3).
- `src/visualizar_partida.py`: Script para assistir o agente jogando.

## 🚀 Como Rodar (Via Docker)

Recomendamos o uso de Docker devido à complexidade das dependências do GFootball.

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Para treinar (Fase Tática Atual):**
   ```bash
   python3 src/04_treino_tatico_wrapper.py
   ```

3. **Para assistir ao Modelo Campeão (1.58):**
   ```bash
   python3 src/visualizar_partida.py
   ```

## 🆘 Ajuda Necessária
Estamos atualmente refinando o `TacticalWrapper` para evitar "Reward Hacking" (onde o bot toca a bola sem objetividade apenas para ganhar pontos). Sugestões são bem-vindas!
