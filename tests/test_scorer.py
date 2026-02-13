from src.scorer import compute_metrics, evaluate_manual_assessment


def test_compute_metrics_basic_shape():
    result = compute_metrics(line_counts=[90, 91, 88, 92, 89], error_count=1)

    assert result["total"] == 450
    assert result["linhas"] == 5
    assert result["nor"] >= 0
    assert "classificacoes" in result
    assert "produtividade" in result["classificacoes"]
    assert "ritmo" in result["classificacoes"]


def test_evaluate_manual_assessment_priority_fields():
    result = evaluate_manual_assessment(
        total_palos=460,
        nor=7.5,
        block_totals=[91, 90, 84, 91, 94],
        margin_left_mm=6.0,
        margin_right_mm=2.0,
        pressure_level="forte",
        organization_level="boa",
    )

    metrics = result["metrics"]
    classes = result["classificacoes"]

    assert metrics["total"] == 460
    assert metrics["nor"] == 7.5
    assert metrics["margem_esquerda_mm"] == 6.0
    assert metrics["margem_direita_mm"] == 2.0
    assert classes["pressao"]["nivel"] == "Forte"
