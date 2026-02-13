import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.pipeline import parse_roi_frac, process_image
from src.scorer import evaluate_manual_assessment, parse_block_totals_text, parse_irregularities_text
from src.ml_models import fuse_ml_with_rules, load_ml_model, predict_ml_classes


def _is_blank(value: str) -> bool:
    return value is None or value.strip() == ""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Palo Analyzer")
        self.geometry("1240x840")
        self.minsize(1100, 760)

        # Paths / execution
        self.image_var = tk.StringVar()
        self.output_var = tk.StringVar(value="output")
        self.roi_var = tk.StringVar(value="0.03,0.14,0.98,0.72")
        self.ml_model_var = tk.StringVar(value="output/ml_models.pkl")
        self.use_ml_var = tk.BooleanVar(value=False)
        self.ml_mode_var = tk.StringVar(value="assist")
        self.ml_threshold_var = tk.StringVar(value="0.75")
        self.swap_lr_margins_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Preencha os campos e clique em 'Gerar Analise'.")

        # Manual inputs (manual > automatic)
        self.m_total = tk.StringVar()
        self.m_nor = tk.StringVar()
        self.m_blocks = tk.StringVar()
        self.m_spacing = tk.StringVar()
        self.m_height = tk.StringVar()
        self.m_line_spacing = tk.StringVar()
        self.m_angle = tk.StringVar()
        self.m_stroke_incl = tk.StringVar()
        self.m_margin_left = tk.StringVar()
        self.m_margin_right = tk.StringVar()
        self.m_margin_top = tk.StringVar()
        self.m_pressure = tk.StringVar(value="nao_informado")
        self.m_stroke_quality = tk.StringVar(value="nao_informado")
        self.m_organization = tk.StringVar(value="nao_informado")
        self.m_irregularities = tk.StringVar()
        self.m_order = tk.StringVar(value="nao_informado")
        self.m_reasoning = tk.StringVar(value="nao_informado")
        self.m_errors = tk.StringVar(value="0")

        # UI output
        self.summary_vars = {
            "total": tk.StringVar(value="-"),
            "nor": tk.StringVar(value="-"),
            "produtividade": tk.StringVar(value="-"),
            "ritmo": tk.StringVar(value="-"),
            "score": tk.StringVar(value="-")
        }
        self.last_output_files = {}
        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Tema neutro para uso profissional.
        self.configure(bg="#f7f8fa")
        style.configure("TFrame", background="#f7f8fa")
        style.configure("TLabelframe", background="#f7f8fa")
        style.configure("TLabelframe.Label", background="#f7f8fa", foreground="#1f2937", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#f7f8fa")

        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text="Analise Palografica Completa",
            font=("Segoe UI", 16, "bold"),
            foreground="#1f2937",
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Fluxo hibrido: imagem opcional + ajustes manuais (manual sempre prevalece)",
            foreground="#4b5563",
        ).pack(anchor="w", pady=(2, 0))

        main = ttk.Panedwindow(root, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=5)
        main.add(right, weight=6)

        self._build_left_panel(left)
        self._build_right_panel(right)

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status_var, anchor="w").pack(fill="x")
        ttk.Label(footer, text="Palo Analyzer", anchor="e", foreground="#4b5563", font=("Segoe UI", 9)).pack(fill="x", pady=(4, 0))

    def _build_left_panel(self, parent):
        source_box = ttk.LabelFrame(parent, text="Fonte de dados")
        source_box.pack(fill="x", pady=(0, 8))

        row = ttk.Frame(source_box, padding=8)
        row.pack(fill="x")
        ttk.Label(row, text="Imagem (opcional):").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.image_var).grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(row, text="Anexar Imagem", command=self.select_image).grid(row=1, column=1)

        ttk.Label(row, text="ROI (para leitura da imagem):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(row, textvariable=self.roi_var, width=30).grid(row=3, column=0, sticky="w")

        ttk.Label(row, text="Pasta de saida:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(row, textvariable=self.output_var).grid(row=5, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(row, text="Selecionar Pasta", command=self.select_output).grid(row=5, column=1)

        ttk.Label(row, text="Modelo ML (.pkl):").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(row, textvariable=self.ml_model_var).grid(row=7, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(row, text="Selecionar Modelo", command=self.select_ml_model).grid(row=7, column=1)

        ml_opts = ttk.Frame(row)
        ml_opts.grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(ml_opts, text="Usar ML na classificacao", variable=self.use_ml_var).pack(side="left")
        ttk.Label(ml_opts, text="Modo:").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            ml_opts,
            textvariable=self.ml_mode_var,
            values=["assist", "hybrid", "override"],
            state="readonly",
            width=10,
        ).pack(side="left")
        ttk.Label(ml_opts, text="Threshold:").pack(side="left", padx=(12, 4))
        ttk.Entry(ml_opts, textvariable=self.ml_threshold_var, width=6).pack(side="left")

        orient_row = ttk.Frame(row)
        orient_row.grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            orient_row,
            text="Trocar margem E/D (imagem espelhada)",
            variable=self.swap_lr_margins_var,
        ).pack(side="left")
        row.columnconfigure(0, weight=1)

        manual_box = ttk.LabelFrame(parent, text="Ajustes manuais (prevalecem sobre automatico)")
        manual_box.pack(fill="both", expand=True)

        form = ttk.Frame(manual_box, padding=8)
        form.pack(fill="both", expand=True)

        # Row 1
        ttk.Label(form, text="Total de palos").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.m_total, width=16).grid(row=1, column=0, sticky="w", padx=(0, 8))

        ttk.Label(form, text="NOR").grid(row=0, column=1, sticky="w")
        ttk.Entry(form, textvariable=self.m_nor, width=16).grid(row=1, column=1, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Blocos (91;90;84;91;94)").grid(row=0, column=2, sticky="w")
        ttk.Entry(form, textvariable=self.m_blocks, width=24).grid(row=1, column=2, sticky="w", padx=(0, 8))

        # Row 2
        ttk.Label(form, text="Dist. entre palos (mm)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_spacing, width=16).grid(row=3, column=0, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Tamanho dos palos (mm)").grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_height, width=16).grid(row=3, column=1, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Dist. entre linhas (mm)").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_line_spacing, width=16).grid(row=3, column=2, sticky="w", padx=(0, 8))

        # Row 3
        ttk.Label(form, text="Direcao linhas (graus)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_angle, width=16).grid(row=5, column=0, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Inclinacao palos (graus)").grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_stroke_incl, width=16).grid(row=5, column=1, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Erros").grid(row=4, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_errors, width=16).grid(row=5, column=2, sticky="w", padx=(0, 8))

        # Row 4
        ttk.Label(form, text="Margem E (mm)").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_margin_left, width=16).grid(row=7, column=0, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Margem D (mm)").grid(row=6, column=1, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_margin_right, width=16).grid(row=7, column=1, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Margem Sup (mm)").grid(row=6, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_margin_top, width=16).grid(row=7, column=2, sticky="w", padx=(0, 8))

        # Row 5
        ttk.Label(form, text="Ordem palos").grid(row=8, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            form,
            textvariable=self.m_order,
            values=["nao_informado", "ordenados", "desordenados"],
            state="readonly",
            width=18,
        ).grid(row=9, column=0, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Raciocinio").grid(row=8, column=1, sticky="w", pady=(8, 0))
        ttk.Combobox(
            form,
            textvariable=self.m_reasoning,
            values=["nao_informado", "normal_ou_superior", "medio_inferior_ou_inferior"],
            state="readonly",
            width=26,
        ).grid(row=9, column=1, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Pressao").grid(row=8, column=2, sticky="w", pady=(8, 0))
        ttk.Combobox(
            form,
            textvariable=self.m_pressure,
            values=["nao_informado", "forte", "media", "leve", "irregular"],
            state="readonly",
            width=16,
        ).grid(row=9, column=2, sticky="w", padx=(0, 8))

        # Row 6
        ttk.Label(form, text="Qualidade tracado").grid(row=10, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            form,
            textvariable=self.m_stroke_quality,
            values=["nao_informado", "reta", "firme", "curva", "frouxa", "descontinua", "interrompida"],
            state="readonly",
            width=20,
        ).grid(row=11, column=0, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Organizacao").grid(row=10, column=1, sticky="w", pady=(8, 0))
        ttk.Combobox(
            form,
            textvariable=self.m_organization,
            values=["nao_informado", "muito boa", "boa", "regular", "ruim", "muito ruim"],
            state="readonly",
            width=20,
        ).grid(row=11, column=1, sticky="w", padx=(0, 8))

        ttk.Label(form, text="Irregularidades (;)").grid(row=10, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(form, textvariable=self.m_irregularities, width=28).grid(row=11, column=2, sticky="w")

        ttk.Label(
            form,
            text="Ex: tremor inicial; gancho superior direito; lacos",
            foreground="#666666",
        ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(4, 0))

        action_row = ttk.Frame(parent)
        action_row.pack(fill="x", pady=(8, 0))
        ttk.Button(action_row, text="Gerar Analise", command=self.run_hybrid_assessment).pack(side="left")
        ttk.Button(action_row, text="Limpar Campos", command=self.clear_manual_fields).pack(side="left", padx=8)
        ttk.Button(action_row, text="Abrir Pasta", command=self.open_output_dir).pack(side="left")

    def _build_right_panel(self, parent):
        summary = ttk.LabelFrame(parent, text="Resumo Executivo")
        summary.pack(fill="x", pady=(0, 8))

        cards = ttk.Frame(summary, padding=8)
        cards.pack(fill="x")
        self._metric_card(cards, "Total", self.summary_vars["total"], 0)
        self._metric_card(cards, "NOR", self.summary_vars["nor"], 1)
        self._metric_card(cards, "Produtividade", self.summary_vars["produtividade"], 2)
        self._metric_card(cards, "Ritmo", self.summary_vars["ritmo"], 3)
        self._metric_card(cards, "Score", self.summary_vars["score"], 4)

        tabs_box = ttk.LabelFrame(parent, text="Detalhamento")
        tabs_box.pack(fill="both", expand=True)

        notebook = ttk.Notebook(tabs_box)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        t1 = ttk.Frame(notebook)
        t2 = ttk.Frame(notebook)
        t3 = ttk.Frame(notebook)
        t4 = ttk.Frame(notebook)

        notebook.add(t1, text="Classificacoes")
        notebook.add(t2, text="Tracos")
        notebook.add(t3, text="Analise Textual")
        notebook.add(t4, text="JSON")

        self.class_tree = ttk.Treeview(t1, columns=("item", "valor"), show="headings")
        self.class_tree.heading("item", text="Item")
        self.class_tree.heading("valor", text="Valor")
        self.class_tree.column("item", width=240, anchor="w")
        self.class_tree.column("valor", width=520, anchor="w")
        self.class_tree.pack(fill="both", expand=True)

        self.trait_tree = ttk.Treeview(t2, columns=("dim", "niv", "regra", "interp"), show="headings")
        self.trait_tree.heading("dim", text="Dimensao")
        self.trait_tree.heading("niv", text="Nivel")
        self.trait_tree.heading("regra", text="Regra")
        self.trait_tree.heading("interp", text="Interpretacao")
        self.trait_tree.column("dim", width=170, anchor="w")
        self.trait_tree.column("niv", width=190, anchor="w")
        self.trait_tree.column("regra", width=100, anchor="w")
        self.trait_tree.column("interp", width=360, anchor="w")
        self.trait_tree.pack(fill="both", expand=True)

        self.notes_text = tk.Text(t3, wrap="word")
        self.notes_text.pack(fill="both", expand=True)

        self.json_text = tk.Text(t4, wrap="none")
        self.json_text.pack(fill="both", expand=True)

        file_row = ttk.Frame(parent)
        file_row.pack(fill="x", pady=(8, 0))
        self.open_overlay_btn = ttk.Button(file_row, text="Abrir Overlay", state="disabled", command=lambda: self.open_result_file("overlay"))
        self.open_overlay_btn.pack(side="left")
        self.open_json_btn = ttk.Button(file_row, text="Abrir JSON", state="disabled", command=lambda: self.open_result_file("json"))
        self.open_json_btn.pack(side="left", padx=8)
        self.open_csv_btn = ttk.Button(file_row, text="Abrir CSV", state="disabled", command=lambda: self.open_result_file("csv"))
        self.open_csv_btn.pack(side="left")

    def _metric_card(self, parent, title, var, col):
        card = ttk.LabelFrame(parent, text=title)
        card.grid(row=0, column=col, padx=(0, 6), sticky="ew")
        ttk.Label(card, textvariable=var, font=("Segoe UI", 12, "bold")).pack(padx=10, pady=6)
        parent.columnconfigure(col, weight=1)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Selecione a imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")],
        )
        if path:
            self.image_var.set(path)

    def select_output(self):
        path = filedialog.askdirectory(title="Selecione a pasta de saida")
        if path:
            self.output_var.set(path)

    def select_ml_model(self):
        path = filedialog.askopenfilename(
            title="Selecione o modelo ML",
            filetypes=[("Modelos pickle", "*.pkl"), ("Todos", "*.*")],
        )
        if path:
            self.ml_model_var.set(path)

    def open_output_dir(self):
        out = self.output_var.get().strip() or "output"
        Path(out).mkdir(parents=True, exist_ok=True)
        os.startfile(out)

    def open_result_file(self, key):
        p = self.last_output_files.get(key)
        if not p or not Path(p).exists():
            messagebox.showwarning("Arquivo", "Arquivo nao disponivel.")
            return
        os.startfile(p)

    def clear_manual_fields(self):
        for var in [
            self.m_total,
            self.m_nor,
            self.m_blocks,
            self.m_spacing,
            self.m_height,
            self.m_line_spacing,
            self.m_angle,
            self.m_stroke_incl,
            self.m_margin_left,
            self.m_margin_right,
            self.m_margin_top,
            self.m_irregularities,
        ]:
            var.set("")
        self.m_pressure.set("nao_informado")
        self.m_stroke_quality.set("nao_informado")
        self.m_organization.set("nao_informado")
        self.m_order.set("nao_informado")
        self.m_reasoning.set("nao_informado")
        self.m_errors.set("0")
        self.ml_mode_var.set("assist")
        self.ml_threshold_var.set("0.75")

    def _clear_result_widgets(self):
        for v in self.summary_vars.values():
            v.set("-")
        for tree in [self.class_tree, self.trait_tree]:
            for item in tree.get_children():
                tree.delete(item)
        self.notes_text.delete("1.0", tk.END)
        self.json_text.delete("1.0", tk.END)
        self.open_overlay_btn.configure(state="disabled")
        self.open_json_btn.configure(state="disabled")
        self.open_csv_btn.configure(state="disabled")

    def _to_optional_float(self, text):
        t = text.strip()
        if not t:
            return None
        return float(t)

    def _to_optional_int(self, text):
        t = text.strip()
        if not t:
            return None
        return int(t)

    def _pick(self, manual, auto):
        return auto if manual is None else manual

    def _pick_text(self, manual_text, auto_text=""):
        return auto_text if _is_blank(manual_text) else manual_text.strip()

    def run_hybrid_assessment(self):
        out = self.output_var.get().strip() or "output"
        Path(out).mkdir(parents=True, exist_ok=True)

        image_path = self.image_var.get().strip()
        auto_metrics = {}
        output_files = {"overlay": None, "json": None, "csv": None}

        # 1) Optional automatic extraction from image
        if image_path:
            try:
                roi_text = self.roi_var.get().strip()
                roi_frac = parse_roi_frac(roi_text) if roi_text else None
                auto = process_image(
                    image_path=image_path,
                    errors=0,
                    roi_frac=roi_frac,
                    output_dir=out,
                    save_artifacts=True,
                    swap_lr_margins=self.swap_lr_margins_var.get(),
                )
                auto_metrics = auto.metrics
                output_files["overlay"] = str(Path(out) / "overlay.jpg")
                output_files["csv"] = str(Path(out) / "contagem_por_linha.csv")
            except Exception as exc:
                messagebox.showerror("Erro imagem", str(exc))
                return

        # 2) Manual inputs override automatic
        try:
            manual_total = self._to_optional_int(self.m_total.get())
            manual_nor = self._to_optional_float(self.m_nor.get())
            manual_blocks = parse_block_totals_text(self.m_blocks.get().strip()) if not _is_blank(self.m_blocks.get()) else None

            total = self._pick(manual_total, auto_metrics.get("total"))
            nor = self._pick(manual_nor, auto_metrics.get("nor"))
            blocks = manual_blocks if manual_blocks is not None else auto_metrics.get("blocos", [])

            if total is None:
                messagebox.showerror("Erro", "Informe o Total de palos manualmente ou anexe imagem para estimar.")
                return

            spacing = self._pick(self._to_optional_float(self.m_spacing.get()), auto_metrics.get("espacamento_medio_mm"))
            height = self._pick(self._to_optional_float(self.m_height.get()), auto_metrics.get("altura_media_palos_mm"))
            line_spacing = self._pick(self._to_optional_float(self.m_line_spacing.get()), auto_metrics.get("distancia_entre_linhas_mm"))
            line_angle = self._pick(self._to_optional_float(self.m_angle.get()), auto_metrics.get("angulo_direcao_linhas_graus"))

            stroke_incl = self._to_optional_float(self.m_stroke_incl.get())
            margin_left = self._to_optional_float(self.m_margin_left.get())
            margin_right = self._to_optional_float(self.m_margin_right.get())
            margin_top = self._to_optional_float(self.m_margin_top.get())
            irregularities = parse_irregularities_text(self.m_irregularities.get().strip())
            errors = int(self.m_errors.get().strip() or "0")

            stroke_incl = self._pick(stroke_incl, auto_metrics.get("angulo_inclinacao_palos_graus"))
            margin_left = self._pick(margin_left, auto_metrics.get("margem_esquerda_mm"))
            margin_right = self._pick(margin_right, auto_metrics.get("margem_direita_mm"))
            margin_top = self._pick(margin_top, auto_metrics.get("margem_superior_mm"))
            pressure_level = (
                self.m_pressure.get().strip()
                if self.m_pressure.get().strip() not in {"", "nao_informado"}
                else auto_metrics.get("classificacoes", {}).get("pressao", {}).get("nivel", "")
            )
            stroke_quality_level = (
                self.m_stroke_quality.get().strip()
                if self.m_stroke_quality.get().strip() not in {"", "nao_informado"}
                else auto_metrics.get("classificacoes", {}).get("qualidade_tracado", {}).get("nivel", "")
            )
            organization_level = (
                self.m_organization.get().strip()
                if self.m_organization.get().strip() not in {"", "nao_informado"}
                else auto_metrics.get("classificacoes", {}).get("organizacao", {}).get("nivel", "")
            )

            result = evaluate_manual_assessment(
                total_palos=int(total),
                nor=nor,
                block_totals=blocks,
                avg_spacing_mm=spacing,
                avg_height_mm=height,
                line_spacing_mm=line_spacing,
                line_direction_angle_deg=line_angle,
                stroke_inclination_angle_deg=stroke_incl,
                margin_left_mm=margin_left,
                margin_right_mm=margin_right,
                margin_top_mm=margin_top,
                pressure_level=pressure_level,
                stroke_quality_level=stroke_quality_level,
                organization_level=organization_level,
                irregularities=irregularities,
                order_pattern=self._pick_text(self.m_order.get(), "nao_informado"),
                reasoning_level=self._pick_text(self.m_reasoning.get(), "nao_informado"),
                error_count=errors,
            )
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            return

        # 3) Audit of source precedence
        source_map = {
            "total_palos": "manual" if manual_total is not None else ("imagem" if auto_metrics else "manual"),
            "nor": "manual" if manual_nor is not None else ("imagem" if auto_metrics.get("nor") is not None else "manual/nao_informado"),
            "blocos": "manual" if manual_blocks is not None else ("imagem" if auto_metrics else "manual/nao_informado"),
            "avg_spacing_mm": "manual" if not _is_blank(self.m_spacing.get()) else ("imagem" if auto_metrics.get("espacamento_medio_mm") is not None else "manual/nao_informado"),
            "avg_height_mm": "manual" if not _is_blank(self.m_height.get()) else ("imagem" if auto_metrics.get("altura_media_palos_mm") is not None else "manual/nao_informado"),
            "line_spacing_mm": "manual" if not _is_blank(self.m_line_spacing.get()) else ("imagem" if auto_metrics.get("distancia_entre_linhas_mm") is not None else "manual/nao_informado"),
            "line_direction_angle_deg": "manual" if not _is_blank(self.m_angle.get()) else ("imagem" if auto_metrics.get("angulo_direcao_linhas_graus") is not None else "manual/nao_informado"),
            "stroke_inclination_angle_deg": "manual" if not _is_blank(self.m_stroke_incl.get()) else ("imagem" if auto_metrics.get("angulo_inclinacao_palos_graus") is not None else "manual/nao_informado"),
            "margin_left_mm": "manual" if not _is_blank(self.m_margin_left.get()) else ("imagem" if auto_metrics.get("margem_esquerda_mm") is not None else "manual/nao_informado"),
            "margin_right_mm": "manual" if not _is_blank(self.m_margin_right.get()) else ("imagem" if auto_metrics.get("margem_direita_mm") is not None else "manual/nao_informado"),
            "margin_top_mm": "manual" if not _is_blank(self.m_margin_top.get()) else ("imagem" if auto_metrics.get("margem_superior_mm") is not None else "manual/nao_informado"),
            "pressure_level": "manual" if self.m_pressure.get().strip() not in {"", "nao_informado"} else ("imagem" if auto_metrics.get("classificacoes", {}).get("pressao") else "manual/nao_informado"),
            "stroke_quality_level": "manual" if self.m_stroke_quality.get().strip() not in {"", "nao_informado"} else ("imagem" if auto_metrics.get("classificacoes", {}).get("qualidade_tracado") else "manual/nao_informado"),
            "organization_level": "manual" if self.m_organization.get().strip() not in {"", "nao_informado"} else ("imagem" if auto_metrics.get("classificacoes", {}).get("organizacao") else "manual/nao_informado"),
            "swap_lr_margins": "usuario" if self.swap_lr_margins_var.get() else "padrao",
        }

        payload = {
            "modo": "hibrido",
            "imagem_anexada": image_path or None,
            "swap_lr_margins": bool(self.swap_lr_margins_var.get()),
            "fontes": source_map,
            "metrics": result["metrics"],
            "classificacoes": result["classificacoes"],
            "tracos_personalidade": result.get("tracos_personalidade", []),
            "irregularidades_avaliadas": result.get("irregularidades_avaliadas", []),
            "regras_aplicadas": result.get("regras_aplicadas", []),
            "observacoes": result.get("observacoes", []),
            "inputs": result.get("inputs", {}),
            "auto_metrics_imagem": auto_metrics if auto_metrics else None,
        }

        if self.use_ml_var.get():
            ml_path = self.ml_model_var.get().strip()
            if not ml_path:
                messagebox.showerror("ML", "Informe o caminho do modelo ML.")
                return
            if not Path(ml_path).exists():
                messagebox.showerror("ML", f"Modelo nao encontrado: {ml_path}")
                return
            try:
                ml_threshold = float(self.ml_threshold_var.get().strip() or "0.75")
                ml_payload = load_ml_model(ml_path)
                ml_preds = predict_ml_classes(payload["metrics"], ml_payload)
                payload["metrics"] = fuse_ml_with_rules(
                    payload["metrics"],
                    ml_preds,
                    mode=self.ml_mode_var.get(),
                    confidence_threshold=ml_threshold,
                )
                payload["ml"] = {
                    "model_path": ml_path,
                    "mode": self.ml_mode_var.get(),
                    "threshold": ml_threshold,
                }
            except Exception as exc:
                messagebox.showerror("ML", f"Falha ao aplicar modelo ML: {exc}")
                return

        json_path = Path(out) / "analise_completa.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        output_files["json"] = str(json_path)
        self.last_output_files = output_files

        self._clear_result_widgets()
        self._render_results(payload)

        self.status_var.set("Analise concluida. Campos manuais tiveram prioridade sobre a leitura automatica.")

    def _render_results(self, payload):
        metrics = payload.get("metrics", {})
        classes = payload.get("classificacoes", {})
        traits = payload.get("tracos_personalidade", [])
        notes = payload.get("observacoes", [])
        auto_quality = metrics.get("auto_quality")

        self.summary_vars["total"].set(str(metrics.get("total", "-")))
        self.summary_vars["nor"].set(str(metrics.get("nor", "-")))
        self.summary_vars["produtividade"].set(classes.get("produtividade", {}).get("nivel", "-"))
        self.summary_vars["ritmo"].set(classes.get("ritmo", {}).get("nivel", "-"))
        self.summary_vars["score"].set(str(metrics.get("score_final", "-")))

        for k, v in classes.items():
            if isinstance(v, dict):
                text = f"{v.get('nivel', '-')} ({v.get('faixa', '-')}) [{v.get('regra_id', 'N/A')}]"
            else:
                text = str(v)
            self.class_tree.insert("", tk.END, values=(k, text))

        for tr in traits:
            self.trait_tree.insert(
                "",
                tk.END,
                values=(
                    tr.get("dimensao", "-"),
                    tr.get("nivel", "-"),
                    tr.get("regra_id", "N/A"),
                    tr.get("interpretacao", "-"),
                ),
            )

        quality_prefix = ""
        if isinstance(auto_quality, dict):
            quality_prefix = (
                f"Confianca automatica: {auto_quality.get('score')} | "
                f"Revisao manual obrigatoria: {auto_quality.get('requires_manual_review')}\n"
            )
            flags = auto_quality.get("flags", [])
            if flags:
                quality_prefix += "Alertas: " + ", ".join(flags) + "\n\n"

        if notes:
            self.notes_text.insert(tk.END, quality_prefix + "\n".join(f"- {n}" for n in notes))
        else:
            base = "Sem observacoes textuais."
            self.notes_text.insert(tk.END, quality_prefix + base)

        self.json_text.insert(tk.END, json.dumps(payload, ensure_ascii=False, indent=2))

        if self.last_output_files.get("overlay") and Path(self.last_output_files["overlay"]).exists():
            self.open_overlay_btn.configure(state="normal")
        if self.last_output_files.get("json") and Path(self.last_output_files["json"]).exists():
            self.open_json_btn.configure(state="normal")
        if self.last_output_files.get("csv") and Path(self.last_output_files["csv"]).exists():
            self.open_csv_btn.configure(state="normal")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
