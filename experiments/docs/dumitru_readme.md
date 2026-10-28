# Pipeline de Deteção de Fogo/Fumo de Dumitru & Spînu (2026) — Implementação e Avaliação no RPi5

Os scripts `experiments/dumitru_positive.py` e `experiments/dumitru_negative.py` implementam a cascata de validação de 5 níveis do paper **arXiv:2607.03131** (secção III-G) e correm-na num **live-stream simulado** no Raspberry Pi 5. O método envolve um detetor YOLOv10 numa sequência de filtros (confiança por classe, tamanho, cor, persistência temporal e cooldown) para reduzir falsos alarmes.

- `dumitru_positive.py` — **latência até ao primeiro alarme** nos 69 vídeos com fogo (6 runs). Resultados em `results/positive_videos/dumitru_latency.xlsx` e `dumitru_latency_details.csv`.
- `dumitru_negative.py` — **falso alarme por clip** (1 = alarme em qualquer ponto do vídeo, 0 = sem alarme) nos 69 vídeos sem fogo (1 run), a métrica do teste FIRESENSE do artigo. Resultados a importar para `results/negative_videos/dumitru_false_alarms.xlsx`.

> **Âmbito:** os scripts implementam apenas o módulo de fogo/fumo do artigo (não o sistema multitarefa completo).

---

## 📄 Paper de Referência

**Título:** A Multi-Task Deep Learning Framework for Real-Time Intelligent Video Surveillance with Temporal Event Validation  
**Autores:** Dumitru & Spînu  
**Publicação:** arXiv preprint arXiv:2607.03131, Julho 2026 (não revisto por pares)  
**Link:** [https://arxiv.org/abs/2607.03131](https://arxiv.org/abs/2607.03131)  
**Código dos autores:** [https://github.com/DumitruEstera/ai-surveillance-system](https://github.com/DumitruEstera/ai-surveillance-system)

O artigo cobre 5 tarefas (rostos, matrículas, armas, ações, fogo/fumo). **Só a secção III-G (fogo/fumo)** é relevante aqui.

### Resumo do método (Secção III-G)

O módulo usa um **YOLOv10** pré-treinado com duas classes (`fire` e `smoke`), embrulhado numa cascata de 5 níveis. Cada candidato produzido pelo YOLO tem de passar, em sequência, por:

1. **Confiança por classe** — fire ≥ 0.40, smoke ≥ 0.55. O fumo é mais exigente por ser mais propenso a falsos positivos com nuvens, vapor e superfícies cinzentas.
2. **Plausibilidade de tamanho** — a caixa deve ocupar entre 0.3% e 85% da área do frame.
3. **Verificação heurística de cor em HSV** — o artigo **não indica os limiares**.
4. **Confirmação temporal por IoU tracking** — a deteção só se confirma após persistir durante "vários frames consecutivos" (o artigo **não indica quantos**).
5. **Cooldown por localização** — evita alarmes repetidos para a mesma fonte.

A cascata distingue o estado `confirmed` (evento real, desenhado no frame) do estado `alert` (momento em que o alarme é disparado).

### Resultados reportados pelo artigo (apenas contexto)

| Teste | Resultado |
| :--- | :--- |
| **Vídeo — FIRESENSE** (49 clips: 24 com fogo, 25 sem fogo) | YOLO cru: deteta 24/24, falsos alarmes **13/25 (52%)**. Pipeline completo: deteta 23/24 (96%), falsos alarmes **1/25 (4%)**. |
| Falsos alarmes por classe (vídeo) | fogo 69% → 6%; fumo 22% → 0% |
| **Imagens — D-Fire** (4306 imagens) | Filtros por frame: falsos alarmes 19,4% → 8,9%, mas deteção **89,7% → 53,6%**. Dos positivos rejeitados: 54% pelo filtro de cor, 24% pela confiança, 22% pelo tamanho. |
| Latência até ao alarme de fogo | ~1,9 s (teste funcional, GPU de portátil) |

**Como ler estes números:**
- O 52% → 4% é **por clip** e sobre apenas 49 clips (13 contra 1 clips com falso alarme).
- O baseline "YOLO cru" dispara se **qualquer frame** do clip detetar, que é o pior caso. Um baseline mais justo seria exigir N frames consecutivos sem os outros filtros.
- O artigo atribui o ganho sobretudo à confirmação multi-frame, mas **não faz ablação por filtro em vídeo** (só compara o pipeline completo com o YOLO cru).
- Os filtros por frame custam recall em imagens estáticas (D-Fire); o artigo justifica que a decisão final cabe à confirmação temporal.

---

## ⚙️ Variáveis e Configurações do Código

### 1. Modelo e execução

| Variável | Valor | Explicação |
| :--- | :---: | :--- |
| `MODEL_PATH` | `models/dumitru/YOLOv10n_best.pt` | Modelo único. **Tem de ser um YOLOv10 treinado com as classes `fire` e `smoke`** (ver nota abaixo). |
| `IMG_SIZE` | `640` | Tamanho de entrada do YOLO. |
| `DEVICE` | `None` | Automático (CPU no RPi5). |
| `N_RUNS` | `6` (positivos) / `1` (negativos) | Repetições por configuração. O pipeline é determinístico, por isso as repetições medem sobretudo a **variabilidade de tempo** (jitter) do Pi. Nos negativos usa-se 1 run, como nos outros métodos. |
| `RUNS_PER_BATCH` | `2` / `1` | Runs executados antes de gravar o checkpoint intermédio. |

**Notas sobre o modelo:**
- Os pesos oficiais `yolov10n.pt` são treinados no COCO e **não têm classes de fogo/fumo**. Com eles, nenhuma deteção seria reconhecida e o script nunca dispararia alarmes.
- O artigo diz apenas "YOLOv10 pré-treinado": **não indica a variante nem os dados de treino**. A escolha da variante *nano* é nossa, para maximizar o throughput no Pi.
- As classes são reconhecidas pelo nome: qualquer nome com "fire"/"flame" vira `fire`, com "smoke" vira `smoke` (função `canonical_class`). Outras classes são ignoradas.
- É preciso uma versão do Ultralytics com suporte a YOLOv10 (ou o fork em que os pesos foram treinados).
- O formato do modelo (`.pt`, ONNX, NCNN) altera muito o tempo de inferência no Pi e, por isso, a latência medida. Fixa um formato e regista-o nos resultados.

### 2. Configurações padrão (`PAPER_DEFAULTS`)

| Variável | Valor | Nível | Origem |
| :--- | :---: | :---: | :--- |
| `fire_conf` | `0.40` | 1 | Paper, Sec. III-G |
| `smoke_conf` | `0.55` | 1 | Paper, Sec. III-G |
| `use_size` | `True` | 2 | Flag da ablação |
| `min_area` | `0.003` | 2 | Paper: 0.3% do frame |
| `max_area` | `0.85` | 2 | Paper: 85% do frame |
| `use_hsv` | `True` | 3 | Flag da ablação |
| `confirm_frames` | `3` | 4 | **Não especificado no paper** — varremos K ∈ {1, 3, 5} |
| `iou_thr` | `0.30` | 4 | **Escolha nossa** (valor comum em trackers por IoU) |
| `max_missed` | `0` | 4 | **Escolha nossa**: frames estritamente consecutivos |
| `cooldown_s` | `10.0` | 5 | **Escolha nossa** |

"Frames consecutivos" significa **frames consecutivos analisados**, não consecutivos no vídeo. Com um modelo lento no Pi, dois frames analisados podem estar vários frames de vídeo afastados, pelo que K frames custam K vezes o tempo de inferência.

### 3. Verificação HSV (`HSV_PARAMS`) — Nível 3

O paper menciona apenas uma *"heuristic colour check in HSV"*, **sem limiares**. Os valores abaixo são **placeholders nossos, não do artigo**. Devem ser calibrados nos teus vídeos.

| Classe | Parâmetros | Significado |
| :--- | :--- | :--- |
| **fire** | `h_max=35, h_min2=170, s_min=70, v_min=120, min_frac=0.10` | Píxeis vermelhos/laranja/amarelos, saturados e brilhantes; pelo menos 10% da caixa. |
| **smoke** | `s_max=80, v_min=50, v_max=240, min_frac=0.50` | Píxeis de baixa saturação (cinzentos/brancos), nem muito escuros nem saturados; pelo menos 50% da caixa. |

Escala OpenCV: H ∈ [0,179], S e V ∈ [0,255]. A caixa é redimensionada para 64×64 antes do teste (custo constante e pequeno). Uma regra de cor de chama discrimina bem o fogo; para o fumo (cinzento) discrimina muito menos.

### 4. Estudo de ablação (`PAPER_CONFIGS`)

O label é o nome da folha nos Excel (`results/README.md`). A run positiva de outubro de 2026 usou os nomes antigos `paper_raw`, ..., `paper_full_k5`, que o `analysis/import_runs.py` traduz.

| Label | Níveis ativos | K | O que mede |
| :--- | :--- | :---: | :--- |
| `raw_K1` | 1 | 1 | Baseline: só confiança por classe |
| `size_K1` | 1 + 2 | 1 | Efeito do filtro de tamanho |
| `size_hsv_K1` | 1 + 2 + 3 | 1 | Efeito do filtro de cor |
| `size_K3` | 1 + 2 + 4 (**sem HSV**) | 3 | Efeito da confirmação temporal, sem cor |
| `full_K3` | 1 a 5 | 3 | Pipeline completa |
| `full_K5` | 1 a 5 | 5 | Exigência temporal mais alta |

O nível 5 (cooldown) está ativo em todas as configurações, mas **não altera a latência medida**, porque cada corrida termina no primeiro alarme.

---

## 🔄 Simulação de Live-Stream

O script usa um **relógio virtual**:

- **Se inferência + pós-processamento demorar MAIS que o período de um frame:** o relógio avança e os frames intermédios são saltados, como num dispositivo que não acompanha o stream.
- **Se demorar MENOS:** o sistema espera pelo frame seguinte. Sem isto, o mesmo frame seria analisado várias vezes e a confirmação temporal (nível 4) contaria um frame como vários.
- O tempo do pós-processamento (parsing, tamanho, HSV, tracking) entra no relógio, porque no Pi esse custo é real.
- **Latência** = (frame visível quando o alarme é disparado − frame de fogo anotado) / FPS.
- **Vídeos negativos:** o vídeo corre até ao primeiro alarme ou até ao fim; qualquer alarme é um falso alarme do clip.

É uma **aproximação razoável**, com limitações: o tempo de leitura/seek do vídeo **não** entra no relógio, e cada corrida termina no primeiro alarme.

---

## 📊 Outputs Gerados

Em `runs/dumitru_positive/` e `runs/dumitru_negative/` (não versionados; juntar a `results/` com `analysis/import_runs.py`).

| Ficheiro | Conteúdo |
| :--- | :--- |
| `results.xlsx` | Positivos: latências, uma folha por configuração, uma linha por run e uma coluna por vídeo. **Célula vazia = sem alarme.** Negativos: 1 = falso alarme, 0 = sem alarme, mais as folhas `summary` (falsos alarmes por configuração) e `details` (frame/segundo do falso alarme e tempos por vídeo). |
| `partial_results.xlsx` | Checkpoint intermédio (sobrescrito a cada batch). |
| `metadata.csv` | Só positivos. Uma linha por run: `experiment, video, run, real_frame, alarm, alarm_frame, early_alarm, fps, latency_seconds, alarm_time_seconds, number_of_inferences, total_inference_time_seconds, postprocess_time_seconds`. Guardado em `results/positive_videos/dumitru_latency_details.csv`. |
| `system_log.csv` | Log a cada 30 min: temperatura, RAM, CPU e clock da GPU do RPi5 (`NaN` fora de um Raspberry Pi). |

**Atenção ao calcular médias:** as células vazias (sem alarme) são falhas de deteção. Se as ignorares, a latência média parece melhor do que é. Reporta também a **taxa de vídeos com alarme** por configuração.

### Métrica `early_alarm`
Se `alarm_frame < real_frame`, a pipeline disparou **antes** do fogo anotado (aparece como latência negativa). Pode ser um falso alarme (algo que não era fogo) ou um frame anotado tarde. Como todos os vídeos contêm fogo, isto só apanha falsos alarmes **anteriores ao início do fogo**.

---

## ⚠️ Âmbito e limitações

- **Falsos alarmes:** o resultado principal do artigo (52% → 4%) usa clips **sem fogo**; é o que o `dumitru_negative.py` mede (alarme em qualquer ponto do clip). Nos positivos, o `early_alarm` é apenas um indicador parcial.
- **Parâmetros que não vêm do artigo:** limiares HSV, K, `iou_thr`, `max_missed` e `cooldown_s`. Os resultados dependem deles.
- **Os resultados dependem do modelo** (variante, dados de treino, formato de exportação) e do estado térmico do Pi. Monitoriza a temperatura e o *throttling*.
- **Teste pequeno no artigo:** 49 clips, sem ablação por filtro em vídeo. Trata os números dos autores como indicativos.

---

## ▶️ Como correr

Caminhos do repositório (ver `experiments/README.md`): modelo em `models/dumitru/` (Git LFS), vídeos em `data/videos/positive|negative` (ou `SAFERFIRE_VIDEOS`), onset frames em `data/onset_frames/<vídeo>.txt` (índice do primeiro frame com fogo, convenção do OpenCV, começa em 0).

```bash
pip install -r experiments/requirements.txt     # é preciso uma versão do Ultralytics com YOLOv10
python experiments/dumitru_positive.py          # saída em runs/dumitru_positive/
python experiments/dumitru_negative.py          # saída em runs/dumitru_negative/
cd analysis
python import_runs.py ../runs/dumitru_negative/results.xlsx ../results/negative_videos/dumitru_false_alarms.xlsx
```