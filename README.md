# Analyzer Palo

Sistema em Python para analise automatizada do Teste Palografico usando visao computacional classica (OpenCV), com suporte a ajustes manuais, regras psicometricas e opcao de assistencia por ML.

## Principais recursos
- Correcao de perspectiva da folha (homografia) e recorte de ROI.
- Binarizacao adaptativa e deteccao/classificacao de palos.
- Extracao automatica de metricas: total, linhas, NOR, inclinacao, margens, espacamentos, pressao estimada, organizacao.
- Regras psicometricas implementadas em codigo com saida estruturada.
- Fluxo hibrido: dados manuais tem prioridade sobre leitura automatica.
- Interface desktop para usuarios leigos (`desktop_app.py`).
- Pipeline de treino/benchmark para ML supervisionado (opcional).

## Estrutura do projeto
```text
Analyzer Palo/
  main.py                         # CLI principal (processa imagem)
  desktop_app.py                  # App desktop (Tkinter)
  config.py                       # Parametros de deteccao/pipeline
  src/
    pipeline.py                   # Pipeline CV + metricas
    preprocessor.py               # Homografia/ROI/binarizacao
    detector.py                   # Deteccao de palos e linhas
    scorer.py                     # Regras e classificacoes
    ml_models.py                  # Treino/predicao/fusao ML
    build_ml_dataset.py           # Gera dataset de features
    train_ml_models.py            # Treina modelos ML
    benchmark_accuracy.py         # Benchmark rules x ML
    run_full_process_examples.py  # Fluxo end-to-end com exemplos reais
  input/                          # Templates e entradas manuais
  output/                         # Artefatos e relatorios gerados
```

## Requisitos
- Python 3.10+
- Windows (desktop app e empacotamento testados em Windows)

## Instalacao
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rapido (CLI)
```powershell
python main.py --image "C:\caminho\folha.jpg" --output-dir output
```

Opcional (ML e margem espelhada):
```powershell
python main.py --image "C:\caminho\folha.jpg" --ml-model "output\ml_models_real_examples.pkl" --ml-mode hybrid --ml-threshold 0.75 --swap-lr-margins
```

## Uso desktop
```powershell
python desktop_app.py
```

No app:
1. Anexe imagem (opcional).
2. Preencha campos manuais (se desejar sobrescrever).
3. Clique em **Gerar Analise Completa**.

## Saidas geradas
Por padrao em `output/`:
- `resultado.json` (pipeline automatico CLI)
- `analise_completa.json` (desktop/hibrido)
- `overlay.jpg`, `aligned.jpg`, `roi.jpg`, `binary.jpg`
- `contagem_por_linha.csv`

## Pipeline de ML (opcional)
1. Gerar dataset:
```powershell
python src/build_ml_dataset.py --input input/ml_labels_template.csv --output output/ml_dataset.csv
```

2. Treinar modelos:
```powershell
python src/train_ml_models.py --dataset output/ml_dataset.csv --model output/ml_models.pkl --report output/ml_train_report.json
```

3. Benchmark:
```powershell
python src/benchmark_accuracy.py --labels input/ml_labels_template.csv --ml-model output/ml_models.pkl --output output/benchmark_report.json
```

4. Fluxo end-to-end com exemplos reais locais:
```powershell
python src/run_full_process_examples.py
```

## Empacotar para outra pessoa (Windows)
```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --windowed --name "AnalyzerPalo" --add-data "input;input" --add-data "output;output" desktop_app.py
```
Executavel em: `dist/AnalyzerPalo/AnalyzerPalo.exe`

## Limitacoes importantes
- Resultado psicometrico depende da qualidade da imagem, calibracao e aderencia ao protocolo de aplicacao.
- Modelos ML atuais sao auxiliares; priorize validacao humana.
- Benchmark com pseudo-rotulos mede consistencia interna, nao validade clinica.

## Etica e uso responsavel
Este software e de apoio tecnico. A interpretacao psicologica final deve ser revisada por profissional habilitado.

## Autor
Ivan Lana
