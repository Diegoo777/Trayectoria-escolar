from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_df(data: bytes, filename: str) -> pd.DataFrame:
    buf = BytesIO(data)
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(buf)
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf, sep=";")
    return pd.read_excel(buf)


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _parse_cycle_key(cycle_str: str):
    parts = str(cycle_str).replace(" SS", "").split("/")
    if len(parts) == 2:
        try:
            a, b = int(parts[0]), int(parts[1])
            if b < a:
                b += 100
            return (a, b)
        except ValueError:
            pass
    return (9999, 9999)


def _fig_json(fig: go.Figure) -> str:
    return fig.to_json()


COLORS = {
    "primary": "#2563eb",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger":  "#dc2626",
    "muted":   "#6b7280",
    "teal":    "#0d9488",
    "orange":  "#ea580c",
}
TEMPLATE = "plotly_white"


# ── data cleaning & deduplication ───────────────────────────────────────────

def _clean(pob: pd.DataFrame):
    pob = pob.copy()

    pob["CALIF"] = pd.to_numeric(pob["CALIF"], errors="coerce")

    if "FECHA" in pob.columns:
        pob["FECHA"] = pd.to_datetime(pob["FECHA"], errors="coerce")

    if "GRCUR" in pob.columns:
        pob["GRCUR"] = pob["GRCUR"].astype("Int64")
    else:
        pob["GRCUR"] = pd.array([0] * len(pob), dtype="Int64")

    if "TIPAS" not in pob.columns:
        pob["TIPAS"] = "FINAL"
    else:
        pob = pob[pob["TIPAS"] != "REVAL"].copy()

    if "OBLIGATORI" not in pob.columns:
        pob["OBLIGATORI"] = "S"
    if "DESCRIP" not in pob.columns:
        pob["DESCRIP"] = pob["MATERIA"].astype(str)
    if "NOMBRE" not in pob.columns:
        pob["NOMBRE"] = pob["ALUMNO"].astype(str)

    orden = {"FINAL": 1, "EXTRA": 2, "REGULA": 3, "REVAL": 4}
    pob["ORDEN_TIPAS"] = pob["TIPAS"].map(orden).fillna(5).astype(int)
    pob["ACRED"] = pob["ACRED"].astype(str).str.strip().str.upper()

    pob_unico = (
        pob.sort_values(
            ["ALUMNO", "MATERIA", "CICLO", "ACRED", "ORDEN_TIPAS"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(subset=["ALUMNO", "MATERIA", "CICLO"], keep="first")
        .copy()
    )
    pob_unico["APROBADO"] = (pob_unico["ACRED"] == "S").astype(int)

    return pob, pob_unico


# ── student trajectories ────────────────────────────────────────────────────

def _trajectories(pob_unico: pd.DataFrame, ciclos_ordenados: list):
    ciclo_orden = {c: i for i, c in enumerate(ciclos_ordenados)}
    pu = pob_unico.copy()
    pu["CICLO_NUM"] = pu["CICLO"].map(ciclo_orden)

    total_oblig = pu[pu["OBLIGATORI"] == "S"]["MATERIA"].nunique()
    if total_oblig == 0:
        total_oblig = pu["MATERIA"].nunique()

    rows = []
    for alumno, g in pu.groupby("ALUMNO"):
        ci = int(g["CICLO_NUM"].min())
        cf = int(g["CICLO_NUM"].max())
        oblig_apro = g[(g["ACRED"] == "S") & (g["OBLIGATORI"] == "S")]["MATERIA"].nunique()
        dur = cf - ci + 1
        rows.append({
            "ALUMNO":             alumno,
            "NOMBRE":             g["NOMBRE"].iloc[0],
            "Ciclo_inicio":       ciclos_ordenados[ci],
            "Ciclo_fin":          ciclos_ordenados[cf],
            "Duracion_ciclos":    dur,
            "Duracion_años":      round(dur / 2, 1),
            "Mat_oblig_aprobadas": oblig_apro,
            "Pct_avance":         round(oblig_apro / total_oblig * 100, 1),
            "Completo":           oblig_apro >= total_oblig,
        })

    return pd.DataFrame(rows), total_oblig


# ── global course stats ──────────────────────────────────────────────────────

def _course_stats(pob_unico: pd.DataFrame):
    tasa = (
        pob_unico.groupby(["MATERIA", "DESCRIP"])
        .agg(
            Total=("APROBADO", "count"),
            Aprobados=("APROBADO", "sum"),
            NP=("CALIF", lambda x: x.isna().sum()),
            GRCUR=("GRCUR", lambda x: int(x[x > 0].mode()[0]) if (x > 0).any() else 0),
        )
        .assign(
            Reprobados=lambda df: df["Total"] - df["Aprobados"],
            Tasa_Apro=lambda df: (df["Aprobados"] / df["Total"] * 100).round(1),
        )
        .sort_values("Tasa_Apro", ascending=False)
        .reset_index()
    )

    aprobados = pob_unico[pob_unico["ACRED"] == "S"].copy()
    tipo = (
        aprobados.groupby(["MATERIA", "DESCRIP", "TIPAS"])
        .size()
        .reset_index(name="n")
    )
    pivot = (
        tipo.pivot_table(
            index=["MATERIA", "DESCRIP"], columns="TIPAS", values="n", fill_value=0
        )
        .reset_index()
    )
    pivot.columns.name = None
    for c in ["FINAL", "EXTRA", "REGULA"]:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot["Total_aprobados"] = pivot["FINAL"] + pivot["EXTRA"] + pivot["REGULA"]
    for c in ["FINAL", "EXTRA", "REGULA"]:
        pivot[f"Pct_{c}"] = (pivot[c] / pivot["Total_aprobados"] * 100).round(1)
    pivot = pivot.sort_values("Pct_FINAL", ascending=False)

    calif = (
        pob_unico.groupby(["MATERIA", "DESCRIP"])
        .agg(Calificacion_promedio=("CALIF", "mean"))
        .round(2)
        .reset_index()
        .sort_values("Calificacion_promedio", ascending=False)
    )

    return tasa, pivot, calif


# ── cohort completion table ─────────────────────────────────────────────────

def _cohort_table(resumen: pd.DataFrame, ciclos: list) -> pd.DataFrame:
    sm = {c: i for i, c in enumerate(ciclos)}
    cohort = (
        resumen.groupby("Ciclo_inicio")
        .agg(
            Total_alumnos=("ALUMNO", "count"),
            Alumnos_completos=("Completo", "sum"),
        )
        .assign(
            Porcentaje_completos=lambda df: (
                df["Alumnos_completos"] / df["Total_alumnos"] * 100
            ).round(1)
        )
        .reset_index()
    )
    cohort["_s"] = cohort["Ciclo_inicio"].map(sm)
    return cohort.sort_values("_s").drop(columns="_s")


# ── global charts ───────────────────────────────────────────────────────────

def _global_charts(resumen, cohort, tasa, pivot, calif, ciclos):
    charts = {}

    total = len(resumen)
    completos = int(resumen["Completo"].sum())

    # Embudo básico
    charts["embudo"] = _fig_json(
        go.Figure(
            go.Funnel(
                y=["Ingresaron", "Completaron materias"],
                x=[total, completos],
                textinfo="value+percent initial",
                marker_color=[COLORS["teal"], COLORS["success"]],
            )
        ).update_layout(title="Embudo General (todas las generaciones)", template=TEMPLATE)
    )

    # Cohort comparison
    fig_coh = go.Figure()
    fig_coh.add_trace(go.Bar(
        x=cohort["Ciclo_inicio"], y=cohort["Total_alumnos"],
        name="Total ingresaron", marker_color="#93c5fd",
    ))
    fig_coh.add_trace(go.Bar(
        x=cohort["Ciclo_inicio"], y=cohort["Alumnos_completos"],
        name="Completaron", marker_color=COLORS["success"],
    ))
    fig_coh.update_layout(
        title="Alumnos por Generación de Ingreso",
        barmode="overlay", xaxis_tickangle=-45,
        yaxis_title="Número de alumnos", template=TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    charts["cohort"] = _fig_json(fig_coh)

    # % completed per generation (line)
    fig_pct = go.Figure(go.Scatter(
        x=cohort["Ciclo_inicio"], y=cohort["Porcentaje_completos"],
        mode="lines+markers+text",
        text=cohort["Porcentaje_completos"].apply(lambda v: f"{v:.0f}%"),
        textposition="top center",
        line_color=COLORS["teal"], marker_color=COLORS["teal"],
    ))
    fig_pct.update_layout(
        title="% de Alumnos que Completan el Programa por Generación",
        yaxis=dict(title="%", range=[0, 105]), xaxis_tickangle=-45,
        template=TEMPLATE,
    )
    charts["pct_ciclo"] = _fig_json(fig_pct)

    # Time to complete
    completados = resumen[resumen["Completo"]].copy()
    if len(completados) > 0:
        conteo = completados["Duracion_ciclos"].value_counts().sort_index()
        colors_t = [
            COLORS["success"] if c == 9 else COLORS["warning"] if c <= 11 else COLORS["danger"]
            for c in conteo.index
        ]
        charts["tiempo"] = _fig_json(
            go.Figure(go.Bar(
                x=[f"{c/2:.1f} años ({c} ciclos)" for c in conteo.index],
                y=conteo.values,
                marker_color=colors_t,
                text=conteo.values, textposition="outside",
            )).update_layout(
                title="Tiempo para Completar el Programa (todas las generaciones)",
                yaxis_title="Número de alumnos", template=TEMPLATE,
            )
        )

    return charts


def _titulacion_global(resumen, df_T, ciclos):
    df_T = _normalize_cols(df_T.copy())
    tit_ids = set(df_T["MATRICULA"].astype(str).str.strip())
    resumen = resumen.copy()
    resumen["Titulado"] = resumen["ALUMNO"].astype(str).str.strip().isin(tit_ids)

    total = len(resumen)
    completos = int(resumen["Completo"].sum())
    titulados = int(resumen["Titulado"].sum())

    charts = {}

    # Full funnel
    charts["embudo"] = _fig_json(
        go.Figure(
            go.Funnel(
                y=["Ingresaron", "Completaron materias", "Se titularon"],
                x=[total, completos, titulados],
                textinfo="value+percent initial",
                marker_color=[COLORS["teal"], COLORS["success"], COLORS["orange"]],
            )
        ).update_layout(title="Embudo General (todas las generaciones)", template=TEMPLATE)
    )

    # Completion vs titulados per generation
    sm = {c: i for i, c in enumerate(ciclos)}
    tit_cic = (
        resumen.groupby("Ciclo_inicio")
        .agg(
            Total=("ALUMNO", "count"),
            Completaron=("Completo", "sum"),
            Titulados=("Titulado", "sum"),
        )
        .assign(
            Pct_Completo=lambda df: (df["Completaron"] / df["Total"] * 100).round(1),
            Pct_Titulado=lambda df: (df["Titulados"] / df["Total"] * 100).round(1),
        )
        .reset_index()
    )
    tit_cic["_s"] = tit_cic["Ciclo_inicio"].map(sm)
    tit_cic = tit_cic.sort_values("_s").drop(columns="_s")

    fig_tit = go.Figure()
    fig_tit.add_trace(go.Scatter(
        x=tit_cic["Ciclo_inicio"], y=tit_cic["Pct_Completo"],
        mode="lines+markers", name="Completaron materias",
        line_color=COLORS["success"], marker_color=COLORS["success"],
    ))
    fig_tit.add_trace(go.Scatter(
        x=tit_cic["Ciclo_inicio"], y=tit_cic["Pct_Titulado"],
        mode="lines+markers", name="Se titularon",
        line_color=COLORS["orange"], marker_color=COLORS["orange"],
        fill="tonexty", fillcolor="rgba(234,88,12,0.12)",
    ))
    fig_tit.update_layout(
        title="Completaron vs. Se Titularon por Generación",
        yaxis=dict(title="%", range=[0, 100]), xaxis_tickangle=-45,
        template=TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    charts["titulacion_ciclo"] = _fig_json(fig_tit)

    return charts, {
        "total_titulados": titulados,
        "pct_titulados": round(titulados / total * 100, 1) if total else 0,
        "pct_tit_de_completos": round(titulados / completos * 100, 1) if completos else 0,
    }


# ── per-generation analysis ─────────────────────────────────────────────────

def analyze_cohort(ciclo_inicio: str, pob_unico: pd.DataFrame, resumen: pd.DataFrame,
                   ciclos_ordenados: list, df_T=None) -> dict:
    """Returns charts + kpis for a single intake generation."""

    gen_set  = set(resumen[resumen["Ciclo_inicio"] == ciclo_inicio]["ALUMNO"])
    gen_data = pob_unico[pob_unico["ALUMNO"].isin(gen_set)].copy()
    gen_res  = resumen[resumen["Ciclo_inicio"] == ciclo_inicio].copy()

    n_total    = len(gen_set)
    n_completos = int(gen_res["Completo"].sum())
    comp_rows   = gen_res[gen_res["Completo"]]

    charts = {}
    kpis = {
        "total":        n_total,
        "completos":    n_completos,
        "pct_completos": round(n_completos / n_total * 100, 1) if n_total else 0,
        "tiempo_prom":  round(float(comp_rows["Duracion_años"].mean()), 1) if len(comp_rows) else None,
        "tiempo_min":   round(float(comp_rows["Duracion_años"].min()), 1) if len(comp_rows) else None,
        "tiempo_max":   round(float(comp_rows["Duracion_años"].max()), 1) if len(comp_rows) else None,
        "titulados":    0,
        "pct_titulados": 0.0,
        "pct_tit_de_completos": 0.0,
    }

    # ── 1. Deserción por semestre ────────────────────────────────────────────
    grcur_v = gen_data[gen_data["GRCUR"].notna() & (gen_data["GRCUR"] > 0)]
    if len(grcur_v) > 0:
        stu_max = (
            grcur_v.groupby("ALUMNO")["GRCUR"]
            .max()
            .reset_index()
            .rename(columns={"GRCUR": "max_sem"})
        )
        max_sem = int(stu_max["max_sem"].max())

        rows_ret = []
        for s in range(1, max_sem + 1):
            n = int((stu_max["max_sem"] >= s).sum())
            rows_ret.append({"sem": s, "label": f"Sem. {s}", "activos": n,
                             "pct": round(n / n_total * 100, 1)})
        ret = pd.DataFrame(rows_ret)
        ret["desertaron"] = [0] + [
            int(ret.loc[i - 1, "activos"] - ret.loc[i, "activos"])
            for i in range(1, len(ret))
        ]
        ret["pct_deser"] = (ret["desertaron"] / n_total * 100).round(1)

        # Dual-axis: bars = active students, line = retention %
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(
            x=ret["label"], y=ret["activos"],
            name="Alumnos activos",
            marker_color=COLORS["teal"],
            text=[f"{v} ({p}%)" for v, p in zip(ret["activos"], ret["pct"])],
            textposition="outside",
        ))
        fig_ret.add_trace(go.Scatter(
            x=ret["label"], y=ret["pct"],
            mode="lines+markers",
            name="% del total",
            yaxis="y2",
            line_color="#1e3a5f", marker_color="#1e3a5f",
        ))
        annotations = []
        for _, row in ret[ret["desertaron"] > 0].iterrows():
            annotations.append(dict(
                x=row["label"],
                y=row["activos"] + n_total * 0.06,
                text=f"<b>-{row['desertaron']}</b>",
                showarrow=False,
                font=dict(size=10, color="#dc2626"),
            ))
        fig_ret.update_layout(
            title=f"Retención de Alumnos por Semestre — Gen. {ciclo_inicio}",
            yaxis=dict(title="Alumnos activos", range=[0, n_total * 1.35]),
            yaxis2=dict(title="% del total", overlaying="y", side="right",
                        range=[0, 135], ticksuffix="%"),
            template=TEMPLATE,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            annotations=annotations,
        )
        charts["desercion"] = _fig_json(fig_ret)

        # Bar: dropout count per semester
        deser_rows = ret[ret["desertaron"] > 0].copy()
        if len(deser_rows) > 0:
            fig_des = go.Figure(go.Bar(
                x=deser_rows["label"],
                y=deser_rows["desertaron"],
                marker_color=[
                    "#ef4444" if p > 10 else "#f97316"
                    for p in deser_rows["pct_deser"]
                ],
                text=[f"{v} ({p}%)" for v, p in zip(
                    deser_rows["desertaron"], deser_rows["pct_deser"]
                )],
                textposition="outside",
            ))
            fig_des.update_layout(
                title=f"Alumnos que Desertaron por Semestre — Gen. {ciclo_inicio}",
                yaxis=dict(
                    title="Número de alumnos",
                    range=[0, int(deser_rows["desertaron"].max()) * 1.4],
                ),
                template=TEMPLATE,
            )
            charts["desercion_barras"] = _fig_json(fig_des)

    # ── 2. Materias fáciles y difíciles ──────────────────────────────────────
    if len(gen_data) > 0:
        gc = (
            gen_data.groupby(["MATERIA", "DESCRIP"])
            .agg(
                Total=("APROBADO", "count"),
                Aprobados=("APROBADO", "sum"),
                avg_calif=("CALIF", "mean"),
                GRCUR=("GRCUR", lambda x: int(x[x > 0].mode()[0]) if (x > 0).any() else 0),
            )
            .assign(Tasa_Apro=lambda df: (df["Aprobados"] / df["Total"] * 100).round(1))
            .reset_index()
        )
        gc = gc[gc["Total"] >= 3].copy()
        gc["avg_calif"] = gc["avg_calif"].round(2)

        apro_g = gen_data[gen_data["ACRED"] == "S"].copy()
        tipo_g = apro_g.groupby(["MATERIA", "TIPAS"]).size().reset_index(name="n")
        tp = tipo_g.pivot_table(index="MATERIA", columns="TIPAS",
                                values="n", fill_value=0).reset_index()
        tp.columns.name = None
        for c in ["FINAL", "EXTRA", "REGULA"]:
            if c not in tp.columns:
                tp[c] = 0
        tp["_t"] = tp["FINAL"] + tp["EXTRA"] + tp["REGULA"]
        tp["Pct_FINAL"] = (tp["FINAL"] / tp["_t"].replace(0, np.nan) * 100).round(1).fillna(0)

        gc = gc.merge(tp[["MATERIA", "Pct_FINAL"]], on="MATERIA", how="left")
        gc["Pct_FINAL"] = gc["Pct_FINAL"].fillna(0)

        def _cls(r):
            # Easy: avg grade ≥ 8 AND ≥80% pass on first attempt
            if r["avg_calif"] >= 8.0 and r["Pct_FINAL"] >= 80:
                return "facil"
            # Hard: <65% pass OR (<60% first attempt AND <80% total approval)
            if r["Tasa_Apro"] < 65 or (r["Pct_FINAL"] < 60 and r["Tasa_Apro"] < 80):
                return "dificil"
            return "normal"

        gc["tipo"] = gc.apply(_cls, axis=1)
        gc = gc.sort_values(["GRCUR", "Tasa_Apro"], ascending=[True, True])

        cmap = {"facil": "#16a34a", "dificil": "#dc2626", "normal": "#64748b"}
        hover = [
            f"{r['DESCRIP'].title()}<br>"
            f"Semestre: {r['GRCUR']}<br>"
            f"Tasa aprobación: {r['Tasa_Apro']}%<br>"
            f"Cal. promedio: {r['avg_calif']}<br>"
            f"Aprobaron en 1er intento: {r['Pct_FINAL']}%<br>"
            f"<b>{r['tipo'].upper()}</b>"
            for _, r in gc.iterrows()
        ]
        ht = max(480, len(gc) * 22 + 150)

        fig_mat = go.Figure()
        fig_mat.add_trace(go.Bar(
            y=gc["DESCRIP"].str.title(),
            x=gc["Tasa_Apro"],
            orientation="h",
            marker_color=gc["tipo"].map(cmap),
            text=gc["Tasa_Apro"].apply(lambda v: f"{v:.0f}%"),
            textposition="inside",
            hovertext=hover, hoverinfo="text",
            showlegend=False,
        ))
        for label, color in [
            ("Fácil — prom. ≥8 y ≥80% en 1er intento", "#16a34a"),
            ("Normal", "#64748b"),
            ("Difícil — <65% aprob. o <60% en 1er intento", "#dc2626"),
        ]:
            fig_mat.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=color, size=10, symbol="square"),
                name=label,
            ))
        fig_mat.update_layout(
            title=f"Materias Fáciles y Difíciles — Gen. {ciclo_inicio}",
            xaxis=dict(title="% aprobación", range=[0, 108]),
            height=ht, template=TEMPLATE,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        charts["materias_gen"] = _fig_json(fig_mat)

    # ── 3. Eficiencia de aprobación por semestre ─────────────────────────────
    apro_g = gen_data[gen_data["ACRED"] == "S"].copy()
    if len(apro_g) > 0:
        apro_g["GRCUR_int"] = apro_g["GRCUR"].fillna(0).astype(int)
        apro_g = apro_g[apro_g["GRCUR_int"] > 0]

        if len(apro_g) > 0:
            sp = (
                apro_g.groupby(["GRCUR_int", "TIPAS"])
                .size()
                .reset_index(name="n")
                .pivot_table(index="GRCUR_int", columns="TIPAS",
                             values="n", fill_value=0)
                .reset_index()
            )
            sp.columns.name = None
            for c in ["FINAL", "EXTRA", "REGULA"]:
                if c not in sp.columns:
                    sp[c] = 0
            sp["_t"] = sp["FINAL"] + sp["EXTRA"] + sp["REGULA"]
            for c in ["FINAL", "EXTRA", "REGULA"]:
                sp[f"Pct_{c}"] = (
                    sp[c] / sp["_t"].replace(0, np.nan) * 100
                ).round(1).fillna(0)
            sp = sp.sort_values("GRCUR_int")
            xlabels = [f"Sem. {s}" for s in sp["GRCUR_int"]]

            fig_ef = go.Figure()
            for col, name, color in [
                ("Pct_FINAL",  "Final",         COLORS["success"]),
                ("Pct_EXTRA",  "Extra",          COLORS["warning"]),
                ("Pct_REGULA", "Regularización", COLORS["danger"]),
            ]:
                fig_ef.add_trace(go.Bar(
                    x=xlabels, y=sp[col], name=name,
                    marker_color=color,
                    text=sp[col].apply(lambda v: f"{v:.0f}%" if v >= 5 else ""),
                    textposition="inside",
                ))
            fig_ef.update_layout(
                title=f"Eficiencia de Aprobación por Semestre — Gen. {ciclo_inicio}",
                barmode="stack",
                yaxis=dict(title="% de aprobados", range=[0, 105], ticksuffix="%"),
                template=TEMPLATE,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            charts["eficiencia_sem"] = _fig_json(fig_ef)

    # ── 4. Titulación para esta generación ────────────────────────────────────
    if df_T is not None:
        dT = _normalize_cols(df_T.copy())
        tit_ids = set(dT["MATRICULA"].astype(str).str.strip())
        gen_str = {str(a).strip() for a in gen_set}
        gen_tit = gen_str & tit_ids
        n_tit = len(gen_tit)

        kpis["titulados"] = n_tit
        kpis["pct_titulados"] = round(n_tit / n_total * 100, 1) if n_total else 0
        kpis["pct_tit_de_completos"] = (
            round(n_tit / n_completos * 100, 1) if n_completos else 0
        )

        if "DESCMODALI" in dT.columns:
            gen_tit_df = dT[dT["MATRICULA"].astype(str).str.strip().isin(gen_tit)]
            if len(gen_tit_df) > 0:
                mod = gen_tit_df["DESCMODALI"].value_counts().reset_index()
                mod.columns = ["Modalidad", "Alumnos"]
                charts["titulacion_gen"] = _fig_json(
                    go.Figure(go.Pie(
                        labels=mod["Modalidad"],
                        values=mod["Alumnos"],
                        hole=0.42,
                        textinfo="percent+label",
                    )).update_layout(
                        title=f"Modalidades de Titulación — Gen. {ciclo_inicio}",
                        template=TEMPLATE,
                    )
                )

    return {"charts": charts, "kpis": kpis}


# ── public entry point ───────────────────────────────────────────────────────

def run_analysis(materias_data: bytes, materias_filename: str,
                 titulaciones_data=None) -> tuple:
    """
    Returns (template_context dict, cache dict).
    template_context goes to render_template; cache is stored server-side.
    """
    pob = _normalize_cols(_load_df(materias_data, materias_filename))

    required = {"ALUMNO", "MATERIA", "CICLO", "ACRED", "CALIF"}
    missing = required - set(pob.columns)
    if missing:
        raise ValueError(
            f"El archivo no tiene las columnas requeridas: {', '.join(sorted(missing))}"
        )

    pob, pob_unico = _clean(pob)
    ciclos = sorted(pob["CICLO"].dropna().unique(), key=_parse_cycle_key)

    resumen, total_oblig = _trajectories(pob_unico, ciclos)
    cohort  = _cohort_table(resumen, ciclos)
    tasa, pivot, calif = _course_stats(pob_unico)

    charts = _global_charts(resumen, cohort, tasa, pivot, calif, ciclos)

    df_T = None
    if titulaciones_data:
        t_data, t_name = titulaciones_data
        df_T = _normalize_cols(_load_df(t_data, t_name))

    completados = resumen[resumen["Completo"]]
    kpis = {
        "total_alumnos":    len(resumen),
        "total_completaron": int(resumen["Completo"].sum()),
        "pct_completaron":  round(resumen["Completo"].mean() * 100, 1),
        "total_materias":   int(pob_unico["MATERIA"].nunique()),
        "materias_obligatorias": int(total_oblig),
        "registros_raw":    len(pob),
        "registros_limpios": len(pob_unico),
        "tiempo_promedio_años": (
            round(completados["Duracion_años"].mean(), 1) if len(completados) else "—"
        ),
    }

    has_titulaciones = df_T is not None
    if has_titulaciones:
        tit_charts, tit_kpis = _titulacion_global(resumen, df_T, ciclos)
        charts.update(tit_charts)
        kpis.update(tit_kpis)

    template_ctx = {
        "charts": charts,
        "kpis": kpis,
        "ciclos": ciclos,
        "has_titulaciones": has_titulaciones,
    }

    cache = {
        "pob_unico": pob_unico,
        "resumen":   resumen,
        "ciclos":    ciclos,
        "df_T":      df_T,
    }

    return template_ctx, cache
