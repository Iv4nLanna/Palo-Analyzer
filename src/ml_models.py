import csv
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "total",
    "linhas",
    "media_por_linha",
    "desvio_padrao",
    "variabilidade_cv",
    "velocidade_linha_seg",
    "erros",
    "score_final",
    "nor",
    "espacamento_medio_mm",
    "altura_media_palos_mm",
    "distancia_entre_linhas_mm",
    "angulo_direcao_linhas_graus",
    "angulo_inclinacao_palos_graus",
    "margem_esquerda_mm",
    "margem_direita_mm",
    "margem_superior_mm",
]

TARGET_MAP = {
    "target_produtividade": ("produtividade", True),
    "target_ritmo": ("ritmo", True),
    "target_qualidade_rendimento": ("qualidade_rendimento", False),
    "target_organizacao": ("organizacao", True),
    "target_direcao_linhas": ("direcao_linhas", True),
    "target_inclinacao_palos": ("inclinacao_palos", True),
    "target_pressao": ("pressao", True),
    "target_qualidade_tracado": ("qualidade_tracado", True),
}


@dataclass
class TrainOutput:
    model_path: str
    report: Dict


def _to_float(value) -> float:
    try:
        if value is None or str(value).strip() == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def extract_feature_vector(metrics: Dict) -> Tuple[List[float], List[str]]:
    vector = [_to_float(metrics.get(k)) for k in FEATURE_NAMES]
    return vector, FEATURE_NAMES


def _load_labeled_rows(dataset_csv: str) -> List[Dict]:
    rows = []
    with open(dataset_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def train_ml_models(dataset_csv: str, model_path: str) -> TrainOutput:
    rows = _load_labeled_rows(dataset_csv)
    if not rows:
        raise RuntimeError("Dataset CSV vazio")

    x_all = []
    for row in rows:
        x_all.append([_to_float(row.get(name)) for name in FEATURE_NAMES])

    models = {}
    report = {}

    for target_col, (target_key, _) in TARGET_MAP.items():
        y = [row.get(target_col, "").strip() for row in rows]
        valid = [(x, t) for x, t in zip(x_all, y) if t]
        if len(valid) < 20:
            report[target_col] = {
                "trained": False,
                "reason": "menos de 20 exemplos rotulados",
                "samples": len(valid),
            }
            continue

        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        class_counts = Counter(ys)
        if len(class_counts) < 2:
            report[target_col] = {
                "trained": False,
                "reason": "apenas uma classe no alvo",
                "samples": len(valid),
                "classes": sorted(class_counts.keys()),
            }
            continue

        use_stratify = all(c >= 2 for c in class_counts.values())

        # split simples para score de referencia
        x_tr, x_te, y_tr, y_te = train_test_split(
            xs,
            ys,
            test_size=0.2,
            random_state=42,
            stratify=ys if use_stratify else None,
        )

        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        clf.fit(x_tr, y_tr)
        y_pred = clf.predict(x_te)

        acc = float(accuracy_score(y_te, y_pred))
        f1m = float(f1_score(y_te, y_pred, average="macro"))

        models[target_col] = clf
        report[target_col] = {
            "trained": True,
            "samples": len(valid),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1m, 4),
            "classes": sorted(set(ys)),
            "target_key": target_key,
            "stratified_split": use_stratify,
        }

    payload = {
        "feature_names": FEATURE_NAMES,
        "target_map": TARGET_MAP,
        "models": models,
        "report": report,
    }

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    return TrainOutput(model_path=model_path, report=report)


def load_ml_model(model_path: str) -> Dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_ml_classes(metrics: Dict, ml_payload: Dict) -> Dict:
    feats = [_to_float(metrics.get(k)) for k in ml_payload["feature_names"]]
    out = {}

    for target_col, clf in ml_payload["models"].items():
        pred = clf.predict([feats])[0]
        conf = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba([feats])[0]
            conf = float(max(probs)) if len(probs) else None
        out[target_col] = {
            "pred": str(pred),
            "confidence": round(conf, 4) if conf is not None else None,
        }

    return out


def apply_ml_predictions(metrics: Dict, ml_preds: Dict, prefer_ml: bool = False) -> Dict:
    cls = metrics.get("classificacoes", {})
    ml_block = {}

    for target_col, pred_data in ml_preds.items():
        target_info = TARGET_MAP.get(target_col)
        if not target_info:
            continue
        class_key, is_dict = target_info
        pred_label = pred_data.get("pred")

        ml_block[class_key] = {
            "nivel_ml": pred_label,
            "confidence": pred_data.get("confidence"),
            "source": "ml_model",
        }

        if prefer_ml:
            if is_dict:
                if class_key not in cls or not isinstance(cls.get(class_key), dict):
                    cls[class_key] = {}
                cls[class_key]["nivel"] = pred_label
                cls[class_key]["regra_id"] = "ML_OVERRIDE"
            else:
                cls[class_key] = pred_label

    metrics["classificacoes"] = cls
    metrics["ml_predictions"] = ml_block
    metrics["ml_prefer_mode"] = bool(prefer_ml)
    return metrics


def fuse_ml_with_rules(
    metrics: Dict,
    ml_preds: Dict,
    mode: str = "assist",
    confidence_threshold: float = 0.75,
) -> Dict:
    """
    mode:
    - assist: nao altera classes; apenas adiciona predicoes ML
    - hybrid: altera classes somente quando confidence >= threshold
    - override: altera classes sempre (equivalente ao prefer_ml)
    """
    mode = (mode or "assist").strip().lower()
    cls = metrics.get("classificacoes", {})
    ml_block = {}

    for target_col, pred_data in ml_preds.items():
        target_info = TARGET_MAP.get(target_col)
        if not target_info:
            continue
        class_key, is_dict = target_info
        pred_label = pred_data.get("pred")
        conf = pred_data.get("confidence")

        ml_block[class_key] = {
            "nivel_ml": pred_label,
            "confidence": conf,
            "source": "ml_model",
            "mode": mode,
        }

        do_apply = False
        if mode == "override":
            do_apply = True
        elif mode == "hybrid":
            do_apply = (conf is not None) and (float(conf) >= float(confidence_threshold))

        if do_apply:
            if is_dict:
                if class_key not in cls or not isinstance(cls.get(class_key), dict):
                    cls[class_key] = {}
                cls[class_key]["nivel"] = pred_label
                cls[class_key]["regra_id"] = "ML_FUSED"
                cls[class_key]["ml_confidence"] = conf
            else:
                cls[class_key] = pred_label

    metrics["classificacoes"] = cls
    metrics["ml_predictions"] = ml_block
    metrics["ml_fusion"] = {
        "mode": mode,
        "confidence_threshold": confidence_threshold,
    }
    return metrics
