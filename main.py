import argparse
import json
from pathlib import Path

from src.ml_models import fuse_ml_with_rules, load_ml_model, predict_ml_classes
from src.pipeline import parse_roi_frac, process_image


def parse_args():
    parser = argparse.ArgumentParser(description="Corretor automatico do teste palografico com OpenCV classico.")
    parser.add_argument("--image", required=True, help="Caminho da imagem da folha preenchida")
    parser.add_argument("--output-dir", default="output", help="Diretorio de saida")
    parser.add_argument("--errors", type=int, default=0, help="Erros manuais para penalizacao no score")
    parser.add_argument(
        "--roi-frac",
        default="",
        help="ROI no formato x1,y1,x2,y2 em fracoes (ex: 0.03,0.14,0.98,0.72)",
    )
    parser.add_argument("--ml-model", default="", help="Arquivo .pkl de modelo ML treinado")
    parser.add_argument(
        "--swap-lr-margins",
        action="store_true",
        help="Troca margem esquerda/direita (use quando imagem estiver espelhada).",
    )
    parser.add_argument(
        "--ml-mode",
        default="assist",
        choices=["assist", "hybrid", "override"],
        help="assist=nao altera classes; hybrid=aplica por confianca; override=sempre aplica ML",
    )
    parser.add_argument("--ml-threshold", type=float, default=0.75, help="Limiar de confianca para modo hybrid")
    return parser.parse_args()


def main():
    args = parse_args()
    roi_frac = parse_roi_frac(args.roi_frac)

    result = process_image(
        image_path=args.image,
        errors=args.errors,
        roi_frac=roi_frac,
        output_dir=args.output_dir,
        save_artifacts=True,
        swap_lr_margins=args.swap_lr_margins,
    )
    metrics = result.metrics

    if args.ml_model:
        ml_payload = load_ml_model(args.ml_model)
        ml_preds = predict_ml_classes(metrics, ml_payload)
        metrics = fuse_ml_with_rules(
            metrics,
            ml_preds,
            mode=args.ml_mode,
            confidence_threshold=args.ml_threshold,
        )

        out_json = Path(args.output_dir) / "resultado.json"
        if out_json.exists():
            data = json.loads(out_json.read_text(encoding="utf-8"))
            data["metrics"] = metrics
            out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Processamento concluido")
    print(f"Total de palos: {metrics['total']}")
    print(f"Linhas detectadas: {metrics['linhas']}")
    print(f"Media por linha: {metrics['media_por_linha']}")
    print(f"NOR: {metrics['nor']}")
    print(f"Classificacao produtividade: {metrics['classificacoes']['produtividade']['nivel']}")
    print(f"Classificacao ritmo: {metrics['classificacoes']['ritmo']['nivel']}")
    print(f"Classificacao distancia: {metrics['classificacoes']['distancia']['nivel']}")
    print(f"Classificacao tamanho palos: {metrics['classificacoes']['tamanho_palos']['nivel']}")
    print(f"Classificacao direcao linhas: {metrics['classificacoes']['direcao_linhas']['nivel']}")
    if metrics.get("ml_predictions"):
        print("ML ativo: predicoes em metrics['ml_predictions']")
        print(f"ML fusion mode: {metrics.get('ml_fusion', {}).get('mode')}")
    print(f"Score final: {metrics['score_final']}")


if __name__ == "__main__":
    main()
