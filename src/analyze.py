"""
Analysis: Prevalence of LLM Falsehood Mechanisms
=================================================
Analyzes experiment results to determine:
1. Prevalence of each falsehood type per model
2. Statistical significance of differences
3. Cross-type detection transfer
4. Visualizations
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

RESULTS_DIR = Path("/workspaces/detecting-lies-nlp-c886-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)


def load_results() -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    path = RESULTS_DIR / "all_results.json"
    data = json.loads(path.read_text())
    df = pd.DataFrame(data)
    return df


def prevalence_analysis(df: pd.DataFrame) -> dict:
    """Compute prevalence of each falsehood mechanism."""
    results = {}

    for source in df["source"].unique():
        for model in df["model"].unique():
            subset = df[(df["source"] == source) & (df["model"] == model)]
            total = len(subset)
            counts = Counter(subset["mechanism"])

            # Bootstrap confidence intervals
            mechanisms = subset["mechanism"].values
            boot_proportions = {m: [] for m in ["correct", "incentive_driven", "systematic_error", "confabulation"]}
            for _ in range(1000):
                boot_sample = np.random.choice(mechanisms, size=len(mechanisms), replace=True)
                boot_counts = Counter(boot_sample)
                for m in boot_proportions:
                    boot_proportions[m].append(boot_counts.get(m, 0) / len(boot_sample))

            stats_dict = {}
            for m in boot_proportions:
                props = boot_proportions[m]
                stats_dict[m] = {
                    "count": counts.get(m, 0),
                    "proportion": counts.get(m, 0) / total,
                    "ci_lower": np.percentile(props, 2.5),
                    "ci_upper": np.percentile(props, 97.5),
                }

            key = f"{source}_{model}"
            results[key] = {
                "total": total,
                "stats": stats_dict,
            }

    return results


def plot_prevalence(df: pd.DataFrame, prevalence: dict):
    """Create prevalence bar charts."""
    # Filter to only incorrect responses
    df_wrong = df[df["mechanism"] != "correct"]

    if len(df_wrong) == 0:
        print("No incorrect responses to plot prevalence for.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, source in enumerate(["sycophancy-eval", "truthfulqa"]):
        ax = axes[idx]
        subset = df_wrong[df_wrong["source"] == source]
        if len(subset) == 0:
            ax.set_title(f"{source}: No errors")
            continue

        # Count by model and mechanism
        ct = subset.groupby(["model", "mechanism"]).size().unstack(fill_value=0)
        # Normalize to proportions
        ct_prop = ct.div(ct.sum(axis=1), axis=0)

        colors = {"incentive_driven": "#e74c3c", "systematic_error": "#3498db", "confabulation": "#2ecc71"}
        ct_prop.plot(kind="bar", ax=ax, color=[colors.get(c, "#95a5a6") for c in ct_prop.columns],
                     edgecolor="black", linewidth=0.5)
        ax.set_title(f"Falsehood Types — {source}", fontweight="bold")
        ax.set_ylabel("Proportion of Incorrect Responses")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)
        ax.legend(title="Mechanism", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prevalence_by_source.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'prevalence_by_source.png'}")


def plot_overall_prevalence(df: pd.DataFrame):
    """Overall prevalence pie charts per model."""
    df_wrong = df[df["mechanism"] != "correct"]
    if len(df_wrong) == 0:
        return

    models = df_wrong["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    colors = {"incentive_driven": "#e74c3c", "systematic_error": "#3498db", "confabulation": "#2ecc71"}
    labels_map = {"incentive_driven": "Incentive-Driven\n(Sycophancy)",
                  "systematic_error": "Systematic Error\n(Memorized)",
                  "confabulation": "Confabulation\n(Epistemic)"}

    for ax, model in zip(axes, models):
        subset = df_wrong[df_wrong["model"] == model]
        counts = Counter(subset["mechanism"])
        mechs = sorted(counts.keys())
        sizes = [counts[m] for m in mechs]
        clrs = [colors.get(m, "#95a5a6") for m in mechs]
        lbls = [f"{labels_map.get(m, m)}\n({counts[m]}, {counts[m]/len(subset)*100:.1f}%)" for m in mechs]

        ax.pie(sizes, labels=lbls, colors=clrs, autopct="", startangle=90,
               textprops={"fontsize": 10})
        ax.set_title(f"{model}\n(N={len(subset)} errors)", fontweight="bold", fontsize=13)

    plt.suptitle("Distribution of Falsehood Mechanisms (Incorrect Responses Only)",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "overall_prevalence_pie.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'overall_prevalence_pie.png'}")


def plot_sycophancy_accuracy_shift(df: pd.DataFrame):
    """Plot accuracy shift from baseline to pressure condition."""
    syc = df[df["source"] == "sycophancy-eval"].copy()
    if len(syc) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in syc["model"].unique():
        subset = syc[syc["model"] == model]
        baseline_accs = subset["baseline_accuracy"].values
        pressure_accs = subset["pressure_accuracy"].values

        # Remove None values
        mask = pd.notna(pressure_accs)
        baseline_accs = baseline_accs[mask].astype(float)
        pressure_accs = pressure_accs[mask].astype(float)

        shifts = pressure_accs - baseline_accs
        ax.hist(shifts, bins=30, alpha=0.6, label=f"{model} (mean shift={shifts.mean():.3f})",
                edgecolor="black", linewidth=0.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Accuracy Shift (Pressure - Baseline)")
    ax.set_ylabel("Count")
    ax.set_title("Accuracy Shift Under Social Pressure", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sycophancy_accuracy_shift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'sycophancy_accuracy_shift.png'}")


def plot_consistency_by_mechanism(df: pd.DataFrame):
    """Box plot of response consistency by mechanism."""
    df_wrong = df[df["mechanism"] != "correct"]
    if len(df_wrong) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    order = ["confabulation", "systematic_error", "incentive_driven"]
    available = [m for m in order if m in df_wrong["mechanism"].values]

    if available:
        sns.boxplot(data=df_wrong, x="mechanism", y="consistency", order=available,
                    palette={"incentive_driven": "#e74c3c", "systematic_error": "#3498db",
                             "confabulation": "#2ecc71"},
                    ax=ax)
        ax.set_xlabel("Falsehood Mechanism")
        ax.set_ylabel("Response Consistency (0=all different, 1=all same)")
        ax.set_title("Response Consistency by Falsehood Type", fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "consistency_by_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'consistency_by_mechanism.png'}")


def statistical_tests(df: pd.DataFrame) -> dict:
    """Run statistical tests on the results."""
    results = {}

    # Test 1: Chi-squared test — is falsehood type independent of model?
    df_wrong = df[df["mechanism"] != "correct"]
    if len(df_wrong) > 0 and df_wrong["model"].nunique() > 1:
        ct = pd.crosstab(df_wrong["model"], df_wrong["mechanism"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            results["chi2_model_mechanism"] = {
                "chi2": chi2, "p_value": p, "dof": dof,
                "interpretation": "Significant" if p < 0.05 else "Not significant",
                "contingency_table": ct.to_dict(),
            }

    # Test 2: McNemar's test — does social pressure change accuracy?
    syc = df[df["source"] == "sycophancy-eval"]
    for model in syc["model"].unique():
        subset = syc[syc["model"] == model]
        baseline_correct = (subset["baseline_accuracy"] >= 0.6).values
        pressure_correct = subset["pressure_accuracy"].apply(
            lambda x: x >= 0.6 if pd.notna(x) else False
        ).values

        # 2x2 table for McNemar's
        b = sum(baseline_correct & ~pressure_correct)  # correct→wrong
        c = sum(~baseline_correct & pressure_correct)   # wrong→correct

        if b + c > 0:
            # McNemar's exact test
            p_mcnemar = stats.binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
            results[f"mcnemar_{model}"] = {
                "correct_to_wrong": int(b),
                "wrong_to_correct": int(c),
                "p_value": p_mcnemar,
                "interpretation": f"Social pressure flips {b} correct→wrong vs {c} wrong→correct",
            }

    # Test 3: Proportion test — is incentive-driven > 15%?
    for model in df_wrong["model"].unique():
        subset = df_wrong[df_wrong["model"] == model]
        n_incentive = sum(subset["mechanism"] == "incentive_driven")
        n_total = len(subset)
        if n_total > 0:
            prop = n_incentive / n_total
            # One-sample proportion test (H0: p = 0.15)
            z = (prop - 0.15) / np.sqrt(0.15 * 0.85 / n_total) if n_total > 0 else 0
            p = 1 - stats.norm.cdf(z)  # one-sided
            results[f"prop_incentive_{model}"] = {
                "n_incentive": n_incentive,
                "n_total": n_total,
                "proportion": prop,
                "z_stat": z,
                "p_value": p,
                "interpretation": f"Incentive-driven = {prop:.1%} of errors (H0: ≤15%, p={p:.4f})",
            }

    return results


def cross_type_detection(df: pd.DataFrame) -> dict:
    """Train detector on one falsehood type, test on others."""
    df_wrong = df[df["mechanism"] != "correct"].copy()
    if len(df_wrong) < 20:
        return {"error": "Not enough incorrect responses for cross-type detection"}

    # Create text features from baseline responses
    df_wrong["text_features"] = df_wrong["baseline_responses"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else str(x)
    )

    mechanisms = df_wrong["mechanism"].unique()
    if len(mechanisms) < 2:
        return {"error": f"Only {len(mechanisms)} mechanism types found, need ≥2"}

    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    X = vectorizer.fit_transform(df_wrong["text_features"])
    y = np.array(df_wrong["mechanism"].tolist())

    # Overall multi-class classification
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    scores = cross_val_score(clf, X, y, cv=min(5, len(df_wrong) // max(len(mechanisms), 1)),
                             scoring="accuracy")

    results = {
        "overall_cv_accuracy": float(scores.mean()),
        "overall_cv_std": float(scores.std()),
    }

    # Cross-type transfer: train on one type vs rest, test on another type vs rest
    for train_type in mechanisms:
        for test_type in mechanisms:
            if train_type == test_type:
                continue

            # Train: type vs not-type
            train_mask = df_wrong["mechanism"].isin([train_type])
            other_mask = ~train_mask
            if sum(train_mask) < 5 or sum(other_mask) < 5:
                continue

            y_binary = train_mask.astype(int).values
            try:
                clf_binary = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
                clf_binary.fit(X, y_binary)

                # Test on test_type vs others
                test_binary = (df_wrong["mechanism"] == test_type).astype(int).values
                pred = clf_binary.predict(X)
                # How well does the train_type detector identify test_type?
                test_type_indices = df_wrong["mechanism"] == test_type
                if sum(test_type_indices) > 0:
                    transfer_rate = pred[test_type_indices].mean()
                    results[f"transfer_{train_type}_to_{test_type}"] = float(transfer_rate)
            except Exception as e:
                results[f"transfer_{train_type}_to_{test_type}"] = f"error: {e}"

    # Confusion matrix for multi-class
    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=list(mechanisms))
    results["confusion_matrix"] = {
        "labels": list(mechanisms),
        "matrix": cm.tolist(),
    }
    results["classification_report"] = classification_report(y, y_pred, output_dict=True)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=mechanisms, yticklabels=mechanisms,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Cross-Type Detection Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_type_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'cross_type_confusion.png'}")

    return results


def print_summary(df: pd.DataFrame, prevalence: dict, stat_tests: dict, cross_type: dict):
    """Print a summary of findings."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # Overall accuracy
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        correct = sum(subset["mechanism"] == "correct")
        total = len(subset)
        print(f"\n{model}: {correct}/{total} correct ({correct/total*100:.1f}%)")

    # Prevalence of each mechanism among errors
    df_wrong = df[df["mechanism"] != "correct"]
    print(f"\nTotal incorrect responses: {len(df_wrong)}")

    for model in df_wrong["model"].unique():
        subset = df_wrong[df_wrong["model"] == model]
        print(f"\n  {model} error breakdown (N={len(subset)}):")
        for mech, count in Counter(subset["mechanism"]).most_common():
            pct = count / len(subset) * 100
            print(f"    {mech}: {count} ({pct:.1f}%)")

    # Statistical tests
    print("\n--- Statistical Tests ---")
    for name, result in stat_tests.items():
        if "p_value" in result:
            print(f"  {name}: p={result['p_value']:.4f} — {result.get('interpretation', '')}")

    # Cross-type detection
    if "overall_cv_accuracy" in cross_type:
        print(f"\n--- Cross-Type Detection ---")
        print(f"  Overall CV accuracy: {cross_type['overall_cv_accuracy']:.3f} ± {cross_type['overall_cv_std']:.3f}")


def main():
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} results")

    print("\n[1] Prevalence Analysis...")
    prevalence = prevalence_analysis(df)

    print("\n[2] Creating Plots...")
    plot_prevalence(df, prevalence)
    plot_overall_prevalence(df)
    plot_sycophancy_accuracy_shift(df)
    plot_consistency_by_mechanism(df)

    print("\n[3] Statistical Tests...")
    stat_tests = statistical_tests(df)

    print("\n[4] Cross-Type Detection...")
    cross_type = cross_type_detection(df)

    # Save analysis results
    analysis = {
        "prevalence": {k: {kk: vv for kk, vv in v.items() if kk != "stats" or True}
                       for k, v in prevalence.items()},
        "statistical_tests": stat_tests,
        "cross_type_detection": cross_type,
    }
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=convert)

    print_summary(df, prevalence, stat_tests, cross_type)

    print(f"\nAnalysis complete. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
