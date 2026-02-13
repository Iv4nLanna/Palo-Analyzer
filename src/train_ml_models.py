import argparse
import json
import sys
from pathlib import Path

# Permite executar via "python src/train_ml_models.py".
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_models import train_ml_models


def parse_args():
    p = argparse.ArgumentParser(description="Treina modelos ML para classificacoes palograficas")
    p.add_argument("--dataset", required=True, help="CSV com features + colunas target_*")
    p.add_argument("--model-out", default="output/ml_models.pkl", help="Arquivo de modelo de saida")
    p.add_argument("--report-out", default="output/ml_train_report.json", help="Arquivo JSON de relatorio")
    return p.parse_args()


def main():
    args = parse_args()
    out = train_ml_models(dataset_csv=args.dataset, model_path=args.model_out)

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(out.report, f, ensure_ascii=False, indent=2)

    print("Treino concluido")
    print(f"Modelo salvo em: {out.model_path}")
    print(f"Relatorio salvo em: {args.report_out}")


if __name__ == "__main__":
    main()
