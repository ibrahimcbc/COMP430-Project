import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
CSV_DIR = os.path.join(OUT_DIR, "csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

DATASETS = ["adult", "bank", "german"]
K_VALUES = [2, 5, 10, 20]
STRATEGIES = ["public", "all"]


def find_latest_ml_csv():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "ml_results_*.csv")))
    if not files:
        raise FileNotFoundError(f"No ml_results_*.csv found in {OUT_DIR}")
    return files[-1]


def load_ml_results():
    path = find_latest_ml_csv()
    print(f"Loading ML results from: {path}")
    df = pd.read_csv(path)
    df['k'] = df['k'].apply(lambda x: np.nan if pd.isna(x) or str(x) == 'NA' else int(x))
    return df


def compute_deltas(df):
    df_with_deltas = df.copy()
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            base = df[(df['dataset'] == dataset) & (df['variant'] == 'baseline') & (df['model'] == model)]
            if base.empty:
                continue
            base_acc = base['accuracy'].values[0]
            base_f1 = base['f1_macro'].values[0]
            
            anon = df_with_deltas[(df_with_deltas['dataset'] == dataset) & 
                                  (df_with_deltas['variant'] == 'anonymized') & 
                                  (df_with_deltas['model'] == model)]
            df_with_deltas.loc[anon.index, 'delta_accuracy'] = base_acc - anon['accuracy'].values
            df_with_deltas.loc[anon.index, 'delta_f1_macro'] = base_f1 - anon['f1_macro'].values
    
    return df_with_deltas


def generate_detailed_tables(df):
    print("Generating detailed per-dataset/model tables...")
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            sub = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()
            if sub.empty:
                continue
            
            sub = sub.sort_values(['variant', 'strategy', 'k'])
            display = []
            
            base = sub[sub['variant'] == 'baseline']
            if not base.empty:
                base_row = base.iloc[0]
                display.append({
                    'Type': 'Baseline',
                    'Strategy': '-',
                    'k': '-',
                    'Accuracy': f"{base_row['accuracy']:.4f}",
                    'F1-Macro': f"{base_row['f1_macro']:.4f}",
                    'Δ Accuracy': '-',
                    'Δ F1': '-'
                })
            
            anon = sub[sub['variant'] == 'anonymized'].sort_values(['strategy', 'k'])
            for _, row in anon.iterrows():
                base_acc = base.iloc[0]['accuracy'] if not base.empty else np.nan
                base_f1 = base.iloc[0]['f1_macro'] if not base.empty else np.nan
                display.append({
                    'Type': 'Anonymized',
                    'Strategy': row['strategy'],
                    'k': int(row['k']) if pd.notna(row['k']) else '-',
                    'Accuracy': f"{row['accuracy']:.4f}",
                    'F1-Macro': f"{row['f1_macro']:.4f}",
                    'Δ Accuracy': f"{base_acc - row['accuracy']:.4f}" if not pd.isna(base_acc) else '-',
                    'Δ F1': f"{base_f1 - row['f1_macro']:.4f}" if not pd.isna(base_f1) else '-'
                })
            
            display_df = pd.DataFrame(display)
            
            fig, ax = plt.subplots(figsize=(12, len(display_df) * 0.4 + 1))
            ax.axis('tight')
            ax.axis('off')
            
            title = f"{dataset.upper()} – {model}"
            ax.text(0.5, 0.98, title, ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
            
            table = ax.table(cellText=display_df.values, colLabels=display_df.columns, 
                            loc='center', cellLoc='center', colLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            out_path = os.path.join(TABLES_DIR, f"{dataset}_{model.lower()}_table.png")
            plt.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def generate_best_settings_table(df):
    print("Generating best-settings summary table")
    rows = []
    
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            base = df[(df['dataset'] == dataset) & (df['variant'] == 'baseline') & (df['model'] == model)]
            if base.empty:
                continue
            
            base_acc = base['accuracy'].values[0]
            base_f1 = base['f1_macro'].values[0]
            
            rows.append({
                'Dataset': dataset.upper(),
                'Model': model,
                'Type': 'Baseline',
                'Strategy': '-',
                'k': '-',
                'Accuracy': f"{base_acc:.4f}",
                'F1-Macro': f"{base_f1:.4f}",
                'Δ Acc': '-',
                'Δ F1': '-'
            })
            
            for strategy in STRATEGIES:
                anon = df[(df['dataset'] == dataset) & (df['variant'] == 'anonymized') & 
                         (df['strategy'] == strategy) & (df['model'] == model)]
                if anon.empty:
                    continue
                
                best = anon.loc[anon['accuracy'].idxmax()]
                rows.append({
                    'Dataset': dataset.upper(),
                    'Model': model,
                    'Type': f'Best {strategy}',
                    'Strategy': strategy,
                    'k': int(best['k']),
                    'Accuracy': f"{best['accuracy']:.4f}",
                    'F1-Macro': f"{best['f1_macro']:.4f}",
                    'Δ Acc': f"{base_acc - best['accuracy']:.4f}",
                    'Δ F1': f"{base_f1 - best['f1_macro']:.4f}"
                })
    
    summary_df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(14, len(summary_df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    ax.text(0.5, 0.98, 'Best Settings Summary', ha='center', va='top', 
            fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    out_path = os.path.join(TABLES_DIR, "best_settings_summary.png")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_accuracy_vs_k_plots(df):
    print("Generating accuracy vs k plots...")
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            sub = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()
            if sub.empty:
                continue
            
            base = sub[sub['variant'] == 'baseline']
            if base.empty:
                continue
            base_acc = base['accuracy'].values[0]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for strategy in STRATEGIES:
                anon = sub[(sub['variant'] == 'anonymized') & (sub['strategy'] == strategy)].sort_values('k')
                if not anon.empty:
                    ax.plot(anon['k'], anon['accuracy'], marker='o', label=strategy, linewidth=2, markersize=6)
            
            ax.axhline(base_acc, color='gray', linestyle='--', linewidth=2, label='baseline')
            ax.set_xlabel('k', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title(f"{dataset.upper()} – {model} Accuracy vs k", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            out_path = os.path.join(FIGURES_DIR, f"{dataset}_{model.lower()}_acc_vs_k.png")
            plt.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def generate_f1_vs_k_plots(df):
    print("Generating F1-macro vs k plots...")
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            sub = df[(df['dataset'] == dataset) & (df['model'] == model)].copy()
            if sub.empty:
                continue
            
            base = sub[sub['variant'] == 'baseline']
            if base.empty:
                continue
            base_f1 = base['f1_macro'].values[0]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for strategy in STRATEGIES:
                anon = sub[(sub['variant'] == 'anonymized') & (sub['strategy'] == strategy)].sort_values('k')
                if not anon.empty:
                    ax.plot(anon['k'], anon['f1_macro'], marker='s', label=strategy, linewidth=2, markersize=6)
            
            ax.axhline(base_f1, color='gray', linestyle='--', linewidth=2, label='baseline')
            ax.set_xlabel('k', fontsize=11)
            ax.set_ylabel('F1-Macro', fontsize=11)
            ax.set_title(f"{dataset.upper()} – {model} F1-Macro vs k", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            out_path = os.path.join(FIGURES_DIR, f"{dataset}_{model.lower()}_f1_vs_k.png")
            plt.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def generate_utility_loss_plots(df):
    print("Generating utility loss (Δ metric) vs k plots...")
    df_deltas = compute_deltas(df)
    
    # Δ Accuracy plots
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            sub = df_deltas[(df_deltas['dataset'] == dataset) & (df_deltas['model'] == model)].copy()
            if sub.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for strategy in STRATEGIES:
                anon = sub[(sub['variant'] == 'anonymized') & (sub['strategy'] == strategy)].sort_values('k')
                if not anon.empty:
                    ax.plot(anon['k'], anon['delta_accuracy'], marker='o', label=strategy, linewidth=2, markersize=6)
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('k', fontsize=11)
            ax.set_ylabel('Accuracy Loss (baseline - anonymized)', fontsize=11)
            ax.set_title(f"{dataset.upper()} – {model} Accuracy Loss vs k", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            out_path = os.path.join(FIGURES_DIR, f"{dataset}_{model.lower()}_acc_loss_vs_k.png")
            plt.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")
    
    # Δ F1-Macro plots
    for dataset in DATASETS:
        for model in ['LogisticRegression', 'RandomForest']:
            sub = df_deltas[(df_deltas['dataset'] == dataset) & (df_deltas['model'] == model)].copy()
            if sub.empty:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            for strategy in STRATEGIES:
                anon = sub[(sub['variant'] == 'anonymized') & (sub['strategy'] == strategy)].sort_values('k')
                if not anon.empty:
                    ax.plot(anon['k'], anon['delta_f1_macro'], marker='s', label=strategy, linewidth=2, markersize=6)
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('k', fontsize=11)
            ax.set_ylabel('F1-Macro Loss (baseline - anonymized)', fontsize=11)
            ax.set_title(f"{dataset.upper()} – {model} F1-Macro Loss vs k", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            out_path = os.path.join(FIGURES_DIR, f"{dataset}_{model.lower()}_f1_loss_vs_k.png")
            plt.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out_path}")


def save_csv_outputs(df):
    print("Saving CSV outputs...")
    
    # Detailed results
    out_path = os.path.join(CSV_DIR, "ml_results_detailed.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    
    # Results with deltas
    df_deltas = compute_deltas(df)
    out_path = os.path.join(CSV_DIR, "ml_results_with_deltas.csv")
    df_deltas.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def generate_visualization(show_tables=True, show_plots=True, show_utility_loss=True):
    print("Starting result evaluation visualization...")
    
    df = load_ml_results()
    save_csv_outputs(df)
    
    if show_tables:
        generate_detailed_tables(df)
        generate_best_settings_table(df)
    
    if show_plots:
        generate_accuracy_vs_k_plots(df)
        generate_f1_vs_k_plots(df)
    
    if show_utility_loss:
        generate_utility_loss_plots(df)
    
    print(f"\nVisualization complete!")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"CSV files saved to: {CSV_DIR}")


if __name__ == '__main__':
    generate_visualization(show_tables=True, show_plots=True)
