# Replicação do Método FLAME (Gragnaniello et al., 2025)

Este script replica e avalia a abordagem **FLAME** (*Fire detection in videos combining a deep neural network with a model-based motion analysis*), um método híbrido de 3 estágios que combina deteção por Deep Learning com análise de movimento baseada em modelo para reduzir drasticamente os falsos positivos em sistemas de videovigilância.

O script foi desenhado para executar **dois regimes de teste distintos** (Offline e Edge) e inclui uma calibração automática do limiar de confiança, garantindo resultados cientificamente robustos e defensáveis.

---

## 📄 Paper de Referência

**Título:** FLAME: fire detection in videos combining a deep neural network with a model-based motion analysis  
**Autores:** Diego Gragnaniello, Antonio Greco, Carlo Sansone, Bruno Vento  
**Publicação:** *Neural Computing and Applications*, 37, 6181–6197 (2025)  
**DOI:** [10.1007/s00521-024-10963-z](https://doi.org/10.1007/s00521-024-10963-z)

**Resumo do Método:**  
O pipeline FLAME opera em três estágios sequenciais:
1. **Stage I (Detecção):** Um detetor YOLOv8s (com cabeçalhos separados para *flame* e *smoke*) gera candidatos a fogo/fumo frame a frame.
2. **Stage II (Filtro de Background):** Um algoritmo de subtração de fundo (Mixture of Gaussians - MOG2) descarta candidatos que se sobreponham a regiões estáticas da imagem, eliminando falsos positivos como reflexos, nevoeiro ou objetos parados.
3. **Stage III (Filtro de Movimento/Tracking):** Um algoritmo de *Centroid Tracking* com uma Máquina de Estados Finita (FSM) analisa a persistência temporal dos candidatos. O alarme só é disparado se o centróide do objeto persistir numa posição coerente com a de um fogo real durante um tempo mínimo ($t_C$), rejeitando objetos em movimento rápido (ex: faróis de carros, pessoas).

---

## ⚙️ Configuração e Variáveis do Código

O script é dividido em configurações gerais e hiperparâmetros específicos dos 3 estágios do FLAME.

### 1. Configurações Gerais e Regimes de Execução
| Variável | Valor no Código | Explicação |
| :--- | :---: | :--- |
| `N_RUNS` | `6` | Número de repetições para o **Regime Edge**. Garante robustez estatística (média ± desvio padrão) face à variabilidade de throughput no Raspberry Pi 5. |
| `RUN_GRID_SEARCH` | `True` | Executa automaticamente uma varredura de limiares de confiança (`Tc` de 0.10 a 0.90) num subconjunto de vídeos *antes* dos testes principais. |
| `RUN_OFFLINE` | `True` | Ativa o **Regime Offline** (validação da implementação). Processa todos os frames sequencialmente. Determinístico (1 run). |
| `RUN_EDGE` | `True` | Ativa o **Regime Edge** (matriz comparativa). Simula o relógio virtual baseado no tempo de inferência real. Não determinístico (6 runs). |
| `GRID_SEARCH_SUBSET`| `10` | Número de vídeos usados apenas para calibrar o `Tc` ideal para o teu modelo YOLO específico. |
| `OFFLINE_SUBSET` | `15` | Número de vídeos usados no Regime Offline para validar a implementação sem gastar horas de processamento desnecessárias. |

### 2. Hiperparâmetros do FLAME (`build_edge_configs` / `build_offline_configs`)
O código constrói as configurações automaticamente com base no `Tc` calibrado. Os parâmetros base são:

| Variável | Valor Padrão | Explicação (Paralelismo com o Paper) |
| :--- | :---: | :--- |
| **`conf`**<br>*(Confidence, $T_c$)* | `0.40`* | Limiar de confiança do YOLOv8s. O paper usa 0.4, mas o script ajusta este valor automaticamente se o *Grid Search* detetar que o teu modelo tem uma curva de confiança diferente. |
| **`t_fg`**<br>*(Foreground Threshold, $T_{fg}$)* | `0.20` | **Stage II:** Percentagem mínima de sobreposição entre a *bounding box* detetada e os píxeis de "primeiro plano" (foreground) do MOG2. Se < 20%, o candidato é descartado como objeto estático. |
| **`tC`**<br>*(Confirmation Time, $t_C$)* | `1.0` s | **Stage III:** Tempo mínimo (em segundos) que um candidato deve ser rastreado com sucesso para mudar do estado *New* para *Confirmed* (disparar alarme). |
| **`tO`**<br>*(Occlusion Time, $t_O$)* | `0.1` s | **Stage III:** Tempo máximo que um candidato pode ficar sem ser detetado (estado *Occluded*) antes de ser permanentemente rejeitado (*Rejected*). |

*\*Nota: O valor de 0.40 é o do paper, mas será substituído pelo valor ótimo encontrado pelo `RUN_GRID_SEARCH`.*

---

## 🔄 Os Dois Regimes de Teste (Conceito Crucial)

Este script avalia o FLAME de duas formas distintas, o que constitui um **contributo científico** para o teu artigo:

1. **Regime OFFLINE (Replicação Fiel):**
   - **Como funciona:** Lê **todos** os frames do vídeo sequencialmente (`cap.read()`). O tempo entre frames (`dt`) é fixo em `1/fps` (ex: 0.033s a 30fps).
   - **Objetivo:** Validar que a tua implementação da FSM e do MOG2 está correta. Neste regime, os parâmetros do paper ($t_C=1.0s$, $t_O=0.1s$) funcionam como anunciado, resultando em latências de poucos segundos e alta precisão.
   - **Execução:** 1 run por configuração (é determinístico).

2. **Regime EDGE (Simulação de Deploy Real no RPi5):**
   - **Como funciona:** Usa um "relógio virtual". O sistema só processa o frame que "chegou" com base no tempo real que o YOLO demorou a inferir o frame anterior (`dt = yolo_time`). Num RPi5, isto pode resultar em ~2 a 5 FPS efetivos.
   - **Objetivo:** Medir o custo real de deploy e comparar com as outras famílias da tua matriz.
   - **O Finding Científico:** Neste regime, como `dt` pode ser ~0.5s, o parâmetro $t_O=0.1s$ do paper torna-se um gargalo. **Uma única falha de deteção faz o track ser rejeitado imediatamente**, forçando o sistema a recomeçar do zero. Isto explica latências altas (15-77s) ou falhas de alarme, quantificando a degradação de métodos temporais em hardware *edge* de baixo throughput.

---

## 🚀 Como Executar

1. **Pré-requisitos:**
   * Ambiente virtual Python ativado.
   * Dependências: `ultralytics`, `opencv-python`, `pandas`, `openpyxl`, `psutil`, `numpy`.
   * Modelo YOLOv8s treinado para fogo/fumo em `FLAME/YOLOv8s_best.pt`.

2. **Execução Recomendada (usando `tmux`):**
   Dado que o Regime Edge com 6 runs pode demorar várias horas, executa dentro de uma sessão `tmux` para evitar interrupções por falha de ligação SSH:
   ```bash
   tmux new -s flame_test
   python launch_simulation_flame.py
   # (Para sair do tmux sem parar o script: pressionar Ctrl+B, depois D)
   # (Para voltar à sessão: tmux attach -t flame_test)