#!/usr/bin/env python3
"""
K-Anonymity Verification Test
Tests k-anonymity for all anonymized datasets and generates PNG reports.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
ANON_DIR = os.path.join(BASE_DIR, "data", "anonymized_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "data")

QI_SETS = {
    'adult': {
        'public': ['age', 'sex', 'race', 'education', 'marital-status', 'native-country'],
        'all': ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    },
    'bank': {
        'public': ['age', 'job', 'marital', 'education'],
        'all': ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
    },
    'german': {
        'public': ['age', 'personal_status', 'job', 'housing'],
        'all': ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
                'savings_status', 'employment', 'installment_rate', 'personal_status',
                'other_parties', 'residence_since', 'property', 'age', 'other_payment_plans',
                'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker']
    }
}

DATASETS = ["adult", "bank", "german"]
K_VALUES = [2, 5, 10, 20]
STRATEGIES = ["public", "all"]

# =============================================================================
# Utility Functions
# =============================================================================

def load_csv_smart(path):
    """Load CSV with flexible delimiter handling"""
    for sep in [';', ',', None]:
        try:
            df = pd.read_csv(path, sep=sep, keep_default_na=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path, keep_default_na=False)


def calculate_star_rate(df, qi_cols):
    """Calculate percentage of '*' cells in QI columns"""
    qi_subset = df[qi_cols]
    total = qi_subset.shape[0] * qi_subset.shape[1]
    star_count = sum((qi_subset[col].astype(str) == '*').sum() for col in qi_cols)
    return (star_count / total * 100) if total > 0 else 0.0


def check_k_anonymity(df, qi_cols, expected_k):
    """Check k-anonymity and return metrics"""
    if df.empty or not qi_cols:
        return 0, 0, False
    
    existing_cols = [col for col in qi_cols if col in df.columns]
    if not existing_cols:
        return 0, 0, False
    
    eq_classes = df.groupby(existing_cols, dropna=False).size()
    k_min = eq_classes.min() if len(eq_classes) > 0 else 0
    num_eq = len(eq_classes)
    passes = (k_min >= expected_k)
    
    return k_min, num_eq, passes


def save_png_table(dataset, results):
    """Save results table as PNG"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    table_data = [[
        r['dataset'], r['strategy'], str(r['k_exp']), str(r['qi_count']),
        str(r['k_min']), r['pass'], str(r['num_eq']), r['star%']
    ] for r in results]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis('tight')
    ax.axis('off')
    
    cols = ['dataset', 'strategy', 'k_exp', 'qi_cnt', 'k_min', 'pass', '#eq', 'star%']
    table = ax.table(cellText=table_data, colLabels=cols, cellLoc='center',
                    loc='center', colWidths=[0.12, 0.12, 0.10, 0.10, 0.10, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.8)
    
    # Style header
    for i in range(len(cols)):
        cell = table[(0, i)]
        cell.set_facecolor('#2E7D32')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style rows
    for i in range(len(table_data)):
        for j in range(len(cols)):
            cell = table[(i+1, j)]
            cell.set_facecolor('#f5f5f5' if i % 2 == 0 else '#ffffff')
            
            if j == 5:  # pass column
                if '✓ PASS' in table_data[i][j]:
                    cell.set_facecolor('#81C784')
                    cell.set_text_props(weight='bold', color='white')
                elif '❌' in table_data[i][j]:
                    cell.set_facecolor('#E57373')
                    cell.set_text_props(weight='bold', color='white')
    
    plt.title(f'K-Anonymity Results - {dataset.upper()}', fontsize=16, fontweight='bold', pad=20)
    
    out_path = os.path.join(OUTPUT_DIR, f'{dataset}_k_anonymity_results.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ PNG: {out_path}")


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("\n" + "="*80)
    print("K-ANONYMITY VERIFICATION TEST")
    print("="*80 + "\n")
    
    # Print QI Sets
    print("QI SET CONFIGURATION:")
    print("-"*80)
    for dataset in DATASETS:
        print(f"\n{dataset.upper()}:")
        for strategy in STRATEGIES:
            qi_list = QI_SETS[dataset][strategy]
            print(f"  {strategy:7s} ({len(qi_list):2d} columns): {qi_list}")
    print("\n" + "="*80 + "\n")
    
    all_results = {d: [] for d in DATASETS}
    
    for dataset in DATASETS:
        print(f"\nTesting {dataset.upper()}...")
        
        for strategy in STRATEGIES:
            for k_val in K_VALUES:
                filename = f"{dataset}_k{k_val}_{strategy}.csv"
                filepath = os.path.join(ANON_DIR, filename)
                
                if not os.path.exists(filepath):
                    print(f"  ✗ {filename}: NOT FOUND")
                    continue
                
                try:
                    df = load_csv_smart(filepath)
                    qi_cols = QI_SETS[dataset][strategy]
                    
                    # Check missing columns
                    missing = [c for c in qi_cols if c not in df.columns]
                    if missing:
                        print(f"  ✗ {filename}: MISSING COLS {missing[:2]}")
                        continue
                    
                    # Calculate metrics
                    k_min, num_eq, passes = check_k_anonymity(df, qi_cols, k_val)
                    star_rate = calculate_star_rate(df, qi_cols)
                    
                    status = "✓ PASS" if passes else "✗ FAIL"
                    print(f"  {status} {filename}: k={k_min}/{k_val}, eq={num_eq}, star={star_rate:.1f}%")
                    
                    all_results[dataset].append({
                        'dataset': dataset,
                        'strategy': strategy,
                        'k_exp': k_val,
                        'qi_count': len(qi_cols),
                        'k_min': k_min,
                        'pass': status,
                        'num_eq': num_eq,
                        'star%': f"{star_rate:.2f}%"
                    })
                
                except Exception as e:
                    print(f"  ✗ {filename}: ERROR - {str(e)[:40]}")
    
    # Generate PNG tables
    print("\n" + "="*80)
    print("Generating PNG tables...")
    for dataset in DATASETS:
        if all_results[dataset]:
            save_png_table(dataset, all_results[dataset])
    
    print("\n" + "="*80)
    print("✓ Test complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
