"""Reproduce every table, figure and number in the paper.

  python run_experiments.py              # full protocol (20 repetitions, ~10 min on 2 cores)
  python run_experiments.py --repeats 3  # quick check

Writes results/: table1_knowledge.csv, table2_performance.csv, runs.csv, summary.json,
fig1_framework.png, fig2_results.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon

import kbdt as K

OUT = Path(__file__).with_name("results")
SIZES = [50, 100, 200, 400, 800, 1600]
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_TEST = 2000
MODELS = {  # name: (knowledge used, factory)
    "Fuzzy-AHP weighted sum":     ("Knowledge only", lambda k, s: K.KnowledgeOnly(k, rule_inference=False)),
    "FKRI (fuzzy AHP + rules)":   ("Knowledge only", lambda k, s: K.KnowledgeOnly(k)),
    "Logistic regression":        ("Data only",      lambda k, s: K.DataDriven("lr", s)),
    "Random forest":              ("Data only",      lambda k, s: K.DataDriven("rf", s)),
    "Ordinal GBM":                ("Data only",      lambda k, s: K.KBDTR(k, False, False, False, random_state=s)),
    "+ monotonicity":             ("Hybrid",         lambda k, s: K.KBDTR(k, False, True, False, random_state=s)),
    "+ KB features":              ("Hybrid",         lambda k, s: K.KBDTR(k, True, False, False, random_state=s)),
    "+ KB features + monotonicity": ("Hybrid",       lambda k, s: K.KBDTR(k, True, True, False, random_state=s)),
    "KB-DTR (proposed)":          ("Hybrid",         lambda k, s: K.KBDTR(k, True, True, True, random_state=s)),
}
PROPOSED = "KB-DTR (proposed)"
COLORS = {"KB-DTR (proposed)": "#2a78d6", "Ordinal GBM": "#eb6834",
          "FKRI (fuzzy AHP + rules)": "#1baf7a", "Logistic regression": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"


# ------------------------------------------------------------------ experiments
def learning_curve_run(seed: int, know: K.Knowledge) -> list[dict]:
    pool, y_pool = K.simulate_enterprises(max(SIZES), seed=1000 + seed)
    Xte, yte = K.simulate_enterprises(N_TEST, seed=5000 + seed)
    rows = []
    for n in SIZES:
        for name, (_, make) in MODELS.items():
            m = K.evaluate(make(know, seed), pool.iloc[:n], y_pool[:n], Xte, yte, mvr=n in (200, 1600))
            rows.append({"experiment": "learning_curve", "seed": seed, "n_train": n, "rho": 0.0, "model": name, **m})
    return rows


def robustness_run(seed: int, know: K.Knowledge, n: int = 200) -> list[dict]:
    Xtr, ytr = K.simulate_enterprises(n, seed=1000 + seed)
    Xte, yte = K.simulate_enterprises(N_TEST, seed=5000 + seed)
    rows = []
    for rho in RHOS:
        k = K.perturb(know, rho, np.random.default_rng(777 + seed))
        for name in ["FKRI (fuzzy AHP + rules)", "Ordinal GBM", PROPOSED]:
            m = K.evaluate(MODELS[name][1](k, seed), Xtr, ytr, Xte, yte, mvr=False)
            rows.append({"experiment": "robustness", "seed": seed, "n_train": n, "rho": rho, "model": name, **m})
    return rows


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get)
    adj, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[k]))
        adj[k] = run
    return adj


def _head(data, n):
    X, y = data
    return X.iloc[:n], y[:n]


# ------------------------------------------------------------------ tables
def table2(lc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    def ms(s):
        return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"
    rows, pvals = [], {}
    for n in (200, 1600):
        d = lc[lc.n_train == n].pivot(index="seed", columns="model", values="macro_f1")
        raw = {m: wilcoxon(d[PROPOSED], d[m]).pvalue for m in MODELS if m != PROPOSED}
        pvals[n] = holm(raw)
    for name, (kind, _) in MODELS.items():
        r = {"Model": name, "Knowledge use": kind}
        for n in (200, 1600):
            g = lc[(lc.n_train == n) & (lc.model == name)]
            star = "" if name == PROPOSED or pvals[n][name] >= 0.05 else "*"
            r[f"Macro-F1 (n={n})"] = ms(g.macro_f1) + star
            r[f"QWK (n={n})"] = ms(g.qwk)
        g = lc[(lc.n_train == 1600) & (lc.model == name)]
        r["AUC (n=1600)"] = ms(g.auc)
        r["MVR % (n=1600)"] = f"{100 * g.mvr.mean():.1f}"
        rows.append(r)
    return pd.DataFrame(rows), pvals


# ------------------------------------------------------------------ figures
def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "savefig.dpi": 300, "savefig.bbox": "tight"})


def fig1(path: Path, know: K.Knowledge):
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 102), ax.set_ylim(0, 56), ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=MUTED, lw=0.7))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=7.3, weight="bold",
                color=INK, linespacing=1.15)
        ax.text(x + w / 2, y + h - (5.6 if "\n" not in title else 9.0), body, ha="center", va="top",
                fontsize=6.5, color=MUTED, linespacing=1.4)

    def arrow(p, q, label=None, off=(0, 1.3), **kw):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=MUTED,
                                     shrinkA=1, shrinkB=1))
        if label:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], label, fontsize=6.2,
                    color=MUTED, **({"ha": "center"} | kw))

    kb, data, ml, out = "#e7f1fb", "#f0efec", "#e5f4e4", "#f9eeda"
    w = know.dim_weights
    X = [0.5, 26.5, 52.5, 78.5]
    W, top, bot, H = 22.5, 32, 2, 22
    box(X[0], top, W, H, "Knowledge base", "Saaty pairwise\njudgements as TFNs\n\nLinguistic rules, e.g.\nIF TM is Low THEN\nreadiness \u2264 Medium", kb)
    box(X[1], top, W, H, "Fuzzy AHP", f"Buckley geometric mean\nexpert aggregation\nconsistency CR < 0.10\n\nw(T, O, E) =\n({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})", kb)
    box(X[2], top, W, H, "Fuzzy rule inference", "27 TOE rules + cap rule\nRuspini partitions\ncentre-average\ndefuzzification\n\n\u2192 FKRI, monotone", kb)
    box(X[0], bot, W, H, "Enterprise\ninnovation data", "9 TOE indicators in [0, 1]\nDI DA AA | RD IO DS TM\nEC CP\n+ firm size, sector", data)
    box(X[2], bot, W, H, "Knowledge-constrained\nordinal GBM", "Frank\u2013Hall thresholds\nmonotone constraints\n\u03b1-arbitration with the\nknowledge model", ml)
    box(X[3], bot, W, top + H - bot, "Decision support",
        "Readiness class and\nP(Low, Medium, High)\n\nTOE diagnostic scores\nand FKRI\n\nWhat-if levers:\nreadiness gain from\n+0.1 on each capability\n\nNever contradicts\nthe knowledge base", out)
    mid_top, mid_bot = top + H / 2, bot + H / 2
    arrow((X[0] + W + 0.6, mid_top), (X[1] - 0.6, mid_top))
    arrow((X[1] + W + 0.6, mid_top), (X[2] - 0.6, mid_top))
    arrow((X[0] + W + 0.6, mid_bot - 4), (X[2] - 0.6, mid_bot - 4), "raw indicators x", off=(0, 1.2))
    arrow((X[0] + W + 0.6, bot + H - 3), (X[2] - 0.6, top + 2), "indicators x", off=(3.5, -3.6), rotation=17)
    arrow((X[2] + W / 2, top - 0.6), (X[2] + W / 2, bot + H + 0.6))
    ax.text(X[2] + W / 2 + 1.2, (top + bot + H) / 2, "FKRI, T, O, E", fontsize=6.2, color=MUTED, va="center")
    arrow((X[2] + W + 0.6, mid_top), (X[3] - 0.6, mid_top))
    arrow((X[2] + W + 0.6, mid_bot), (X[3] - 0.6, mid_bot))
    fig.savefig(path)
    plt.close(fig)


def fig2(path: Path, lc: pd.DataFrame, rb: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})
    show = ["KB-DTR (proposed)", "Ordinal GBM", "Logistic regression", "FKRI (fuzzy AHP + rules)"]
    labels = {"KB-DTR (proposed)": "KB-DTR", "Ordinal GBM": "Ordinal GBM",
              "Logistic regression": "Logistic reg.", "FKRI (fuzzy AHP + rules)": "FKRI"}
    for ax, df, x, xlab, title, models in [
            (axes[0], lc, "n_train", "Training firms (log scale)", "(a) Data efficiency", show),
            (axes[1], rb, "rho", "Share of corrupted knowledge ρ", "(b) Robustness to wrong knowledge (n = 200)",
             ["KB-DTR (proposed)", "Ordinal GBM", "FKRI (fuzzy AHP + rules)"])]:
        ends = []
        for m in models:
            g = df[df.model == m].groupby(x).macro_f1
            mu, ci = g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(g.count())
            lw = 2.0 if m == PROPOSED else 1.4
            ax.fill_between(mu.index, mu - ci, mu + ci, color=COLORS[m], alpha=0.15, lw=0)
            ls = "--" if (x == "rho" and m == "Ordinal GBM") else "-"
            ax.plot(mu.index, mu, color=COLORS[m], lw=lw, ls=ls, marker="o", ms=3.2,
                    label=labels[m] + (" (no knowledge)" if ls == "--" else ""), zorder=3 if m == PROPOSED else 2)
            ends.append([mu.iloc[-1], labels[m], COLORS[m]])
        ax.set_xlabel(xlab), ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", color="#e4e3df", lw=0.6), ax.set_axisbelow(True)
        if x == "n_train":
            ax.set_xscale("log"), ax.set_xticks(SIZES), ax.set_xticklabels(SIZES), ax.minorticks_off()
        else:
            ax.set_xticks(RHOS)
    axes[0].set_ylabel("Macro-F1 on 2,000 test firms")
    axes[0].legend(frameon=False, fontsize=7.4, loc="lower right")
    axes[1].legend(frameon=False, fontsize=7.4, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--from-runs", action="store_true", help="rebuild tables/figures from results/runs.csv")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    know, t1, cr = K.elicit()

    t1.assign(**{"Crisp weight": t1["Crisp weight"].round(3)}).to_csv(OUT / "table1_knowledge.csv", index=False)
    if a.from_runs:
        df = pd.read_csv(OUT / "runs.csv")
        a.repeats = int(df.seed.nunique())
    else:
        runs = Parallel(n_jobs=a.jobs)(delayed(learning_curve_run)(s, know) for s in range(a.repeats))
        runs += Parallel(n_jobs=a.jobs)(delayed(robustness_run)(s, know) for s in range(a.repeats))
        df = pd.DataFrame([r for rr in runs for r in rr])
        df.to_csv(OUT / "runs.csv", index=False)
    lc, rb = df[df.experiment == "learning_curve"], df[df.experiment == "robustness"]

    t2, pvals = table2(lc)
    t2.to_csv(OUT / "table2_performance.csv", index=False)
    fig1(OUT / "fig1_framework.png", know)
    fig2(OUT / "fig2_results.png", lc, rb)

    # decision-support example used in the paper
    X, y = K.simulate_enterprises(1600, seed=1000)
    model = K.KBDTR(know).fit(X, y)
    Xe, _ = K.simulate_enterprises(50, seed=4242)
    i = int(np.argmin(np.abs(model.expected_level(Xe) - 0.6)))
    firm = Xe.iloc[[i]].copy()
    P = model.predict_proba(firm)[0]
    fis = K.FuzzyReadinessInference(know)
    fkri, D = fis(K._impute(K._criteria(firm), model.medians_))
    lev = K.improvement_levers(model.expected_level, firm)

    # arbitration weight alpha chosen by KB-DTR at each training size (10 seeds)
    alpha = {n: float(np.mean([K.KBDTR(know, random_state=s).fit(*_head(K.simulate_enterprises(max(SIZES), 1000 + s), n)).alpha_
                               for s in range(10)])) for n in SIZES}
    mean = lambda d, m, col: float(d[d.model == m][col].mean())
    summary = {
        "repeats": a.repeats, "n_test": N_TEST,
        "consistency_ratios": cr,
        "dimension_weights": dict(zip(K.DIMENSIONS, map(float, know.dim_weights))),
        "global_weights": know.global_weights.round(4).to_dict(),
        "true_beta": K.TRUE_BETA,
        "holm_p_macro_f1": {str(n): {k: float(v) for k, v in p.items()} for n, p in pvals.items()},
        "macro_f1_by_n": lc.groupby(["model", "n_train"]).macro_f1.mean().unstack().round(4).to_dict("index"),
        "robustness_macro_f1": rb.groupby(["model", "rho"]).macro_f1.mean().unstack().round(4).to_dict("index"),
        "alpha_by_n": alpha,
        "mvr_n1600": {m: mean(lc[lc.n_train == 1600], m, "mvr") for m in MODELS},
        "example_firm": {"features": {k: (None if pd.isna(v) else (v if isinstance(v, str) else round(float(v), 3)))
                                      for k, v in firm.iloc[0].items()},
                         "T": float(D[0, 0]), "O": float(D[0, 1]), "E": float(D[0, 2]), "FKRI": float(fkri[0]),
                         "proba": P.round(3).tolist(), "levers": lev.round(4).to_dict("records")},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(t1.drop(columns="Indicator").to_string(index=False))
    print(t2.to_string(index=False))
    print(json.dumps({k: summary[k] for k in ["macro_f1_by_n", "robustness_macro_f1", "example_firm"]}, indent=1))


if __name__ == "__main__":
    main()
