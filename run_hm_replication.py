#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heterogeneous causal effect of acquisition-category on customer retention (F2) and LTV
— public replication on the H&M Personalized Fashion Recommendations dataset (Kaggle).

Method (mirrors the apparel-EC study this replicates):
  - Treatment T(g): a customer's FIRST purchase category (product_group) == g, one-vs-rest.
  - Outcomes:  F2  = repeat purchase within `--f2-window` days of first purchase (binary)
               LTV = repeat revenue within the window (continuous, winsorized 99%).
  - Estimator: LinearDML (econml), X=None, W = customer covariates (double machine learning).
               (We deliberately keep treatment-defining item attributes OUT of W to avoid the
                positivity/leakage pitfall: see README.)
  - Heterogeneity: per-category ATE, then Spearman(mean first-purchase price, ATE)
               -> tests whether a single product characteristic (price tier) governs retention.

Usage:
  python run_hm_replication.py --data-dir ./data --out-dir ./output
Requires the three Kaggle CSVs in --data-dir: transactions_train.csv, customers.csv, articles.csv
"""
import os, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import lightgbm as lgb
from econml.dml import LinearDML


def parse_args():
    p = argparse.ArgumentParser(description='H&M heterogeneous retention/LTV replication')
    p.add_argument('--data-dir', default='./data', help='dir with transactions_train.csv, customers.csv, articles.csv')
    p.add_argument('--out-dir', default='./output')
    p.add_argument('--sample-frac', type=float, default=0.2, help='customer sampling fraction (0-1)')
    p.add_argument('--f2-window', type=int, default=90, help='retention window in days')
    p.add_argument('--cohort-start', default='2019-01-01')
    p.add_argument('--cohort-end', default='2020-06-22', help='last first-purchase date with full follow-up')
    p.add_argument('--min-group-n', type=int, default=1000, help='min treated N per category to estimate')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


LGB = dict(n_estimators=200, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, min_child_samples=50, n_jobs=-1, verbose=-1)
W_BASE = ['age', 'age_missing', 'club_member_status', 'fashion_news_frequency',
          'first_amount', 'n_items', 'first_channel', 'first_month']


def build_cohort(a):
    """Read data, derive first-purchase cohort with F2/LTV outcomes and customer covariates."""
    print('[1/4] loading customers & articles ...', flush=True)
    cust = pd.read_csv(os.path.join(a.data_dir, 'customers.csv'),
                       usecols=['customer_id', 'club_member_status', 'fashion_news_frequency', 'age'])
    art = pd.read_csv(os.path.join(a.data_dir, 'articles.csv'),
                      usecols=['article_id', 'product_group_name'], dtype={'article_id': str})
    art_grp = dict(zip(art['article_id'], art['product_group_name']))

    rng = np.random.RandomState(a.seed)
    samp = set(cust['customer_id'].sample(frac=a.sample_frac, random_state=a.seed))
    print(f'      sampled customers: {len(samp):,}', flush=True)

    print('[2/4] loading transactions (chunked, filtered to sample) ...', flush=True)
    parts = []
    for ch in pd.read_csv(os.path.join(a.data_dir, 'transactions_train.csv'), chunksize=3_000_000,
                          usecols=['t_dat', 'customer_id', 'article_id', 'price', 'sales_channel_id'],
                          dtype={'article_id': str}):
        parts.append(ch[ch['customer_id'].isin(samp)])
    tx = pd.concat(parts, ignore_index=True); del parts
    tx['t_dat'] = pd.to_datetime(tx['t_dat'])
    tx['pgroup'] = tx['article_id'].map(art_grp)
    print(f'      transactions kept: {len(tx):,}', flush=True)

    print('[3/4] building first-purchase cohort & outcomes ...', flush=True)
    first = tx.groupby('customer_id')['t_dat'].min().rename('first_date')
    tx = tx.join(first, on='customer_id')
    fd = tx[tx['t_dat'] == tx['first_date']].copy()
    anchor = fd.loc[fd.groupby('customer_id')['price'].idxmax(), ['customer_id', 'pgroup', 'sales_channel_id']]
    anchor = anchor.rename(columns={'pgroup': 'first_group', 'sales_channel_id': 'first_channel'}).set_index('customer_id')
    agg = fd.groupby('customer_id').agg(first_amount=('price', 'sum'), n_items=('price', 'size'))
    coh = anchor.join(agg).join(first)
    coh = coh[(coh['first_date'] >= a.cohort_start) & (coh['first_date'] <= a.cohort_end)]
    coh = coh.merge(cust.set_index('customer_id'), left_index=True, right_index=True, how='left')

    tx = tx.join(coh[['first_date']].rename(columns={'first_date': 'fd2'}), on='customer_id')
    t2 = tx.dropna(subset=['fd2'])
    d = (t2['t_dat'] - t2['fd2']).dt.days
    rep = t2[(d > 0) & (d <= a.f2_window)]
    coh = coh.join(rep.groupby('customer_id').size().rename('n_rep'))
    coh = coh.join(rep.groupby('customer_id')['price'].sum().rename('ltv'))
    coh['F2'] = (coh['n_rep'].fillna(0) > 0).astype(int)
    cap = coh['ltv'].fillna(0).quantile(0.99)
    coh['ltv'] = coh['ltv'].fillna(0.0).clip(upper=cap)

    # covariates W (customer/context only; NOT the category that defines T)
    coh['age_missing'] = coh['age'].isna().astype(int)
    coh['age'] = coh['age'].fillna(coh['age'].median())
    coh['first_month'] = pd.to_datetime(coh['first_date']).dt.month.astype(float)
    for c in ['club_member_status', 'fashion_news_frequency']:
        coh[c] = coh[c].astype('category').cat.codes.astype(float)
    coh['first_channel'] = coh['first_channel'].astype(float)
    coh[W_BASE] = coh[W_BASE].astype(float)
    print(f'      cohort N={len(coh):,}  F2 baseline={coh["F2"].mean():.3f}', flush=True)
    return coh


def per_group_effects(coh, outcome, base, min_n, seed):
    """One-vs-rest LinearDML ATE per product_group for the given outcome column."""
    rows = []
    W = coh[W_BASE].values
    for g, n in coh['first_group'].value_counts().items():
        if n < min_n:
            continue
        T = (coh['first_group'] == g).astype(int).values
        Y = coh[outcome].astype(float).values
        est = LinearDML(model_y=lgb.LGBMRegressor(random_state=seed, **LGB),
                        model_t=lgb.LGBMClassifier(random_state=seed, **LGB),
                        discrete_treatment=True, cv=3, random_state=seed)
        est.fit(Y, T, X=None, W=W)
        ate = float(est.ate(X=None)); lo, hi = est.ate_interval(X=None, alpha=0.05)
        rows.append({'product_group': g, 'N': int(T.sum()), 'ATE': ate, 'CI_lo': float(lo), 'CI_hi': float(hi),
                     'lift_pct': 100 * ate / base, 'sig5': not (lo <= 0 <= hi),
                     'mean_first_price': float(coh.loc[coh['first_group'] == g, 'first_amount'].mean())})
        print(f'      {g[:26]:<26} N={int(T.sum()):>6} ATE={ate:+.4f} [{float(lo):+.4f},{float(hi):+.4f}]', flush=True)
    return pd.DataFrame(rows).sort_values('ATE', ascending=False).reset_index(drop=True)


def main():
    a = parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    coh = build_cohort(a)
    baseF2 = coh['F2'].mean(); baseLTV = coh['ltv'].mean()

    print('[4/4] estimating per-category effects (LinearDML one-vs-rest) ...', flush=True)
    print('  -- F2 --', flush=True)
    f2 = per_group_effects(coh, 'F2', baseF2, a.min_group_n, a.seed)
    print('  -- LTV --', flush=True)
    ltv = per_group_effects(coh, 'ltv', baseLTV, a.min_group_n, a.seed)

    f2.to_csv(os.path.join(a.out_dir, 'hm_f2_by_category.csv'), index=False)
    ltv.to_csv(os.path.join(a.out_dir, 'hm_ltv_by_category.csv'), index=False)

    rho_f2 = spearmanr(f2['mean_first_price'], f2['ATE']).correlation
    m = f2.merge(ltv[['product_group', 'ATE']].rename(columns={'ATE': 'ltv_ate'}), on='product_group')
    rho_f2ltv = spearmanr(m['lift_pct'], m['ltv_ate']).correlation

    # Figure 1: F2 effect vs price tier (the key heterogeneity result)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(f2['mean_first_price'], f2['lift_pct'], s=70, c='#2166ac', edgecolors='black')
    for r in f2.itertuples():
        ax.annotate(r.product_group, (r.mean_first_price, r.lift_pct), fontsize=7, xytext=(3, 3),
                    textcoords='offset points')
    if len(f2) >= 2:
        z = np.polyfit(f2['mean_first_price'], f2['lift_pct'], 1)
        xs = np.linspace(f2['mean_first_price'].min(), f2['mean_first_price'].max(), 50)
        ax.plot(xs, np.polyval(z, xs), ls='--', color='#888')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Mean first-purchase price (= "scale"/tier proxy)')
    ax.set_ylabel('F2 effect (Lift%)')
    ax.set_title(f'H&M: acquisition price tier governs retention\nSpearman(price, effect) = {rho_f2:+.2f}')
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'hm_price_vs_f2.png'), dpi=150, bbox_inches='tight')

    # Figure 2: forest of per-category F2 effects
    F = f2.sort_values('lift_pct'); fig, ax = plt.subplots(figsize=(9, 6))
    for i, r in enumerate(F.itertuples()):
        col = '#1b7837' if (r.sig5 and r.lift_pct > 0) else ('#b2182b' if r.sig5 else '#999')
        ax.errorbar(r.lift_pct, i, xerr=[[r.lift_pct - 100 * r.CI_lo / baseF2], [100 * r.CI_hi / baseF2 - r.lift_pct]],
                    fmt='o', color=col, ecolor=col, capsize=3, markersize=6)
    ax.axvline(0, color='k', ls='--', lw=1.2); ax.set_yticks(range(len(F)))
    ax.set_yticklabels([f'{r.product_group[:28]} (N={r.N})' for r in F.itertuples()], fontsize=8)
    ax.set_xlabel('F2 Lift% (LinearDML, one-vs-rest)')
    ax.set_title('H&M: F2 effect by first-purchase category\n(green=sig+, red=sig-, gray=n.s.)')
    ax.grid(axis='x', alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'hm_f2_forest.png'), dpi=150, bbox_inches='tight')

    # Figure 3: F2 effect vs LTV effect by category
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(m['lift_pct'], m['ltv_ate'], s=70, c='#444444', edgecolors='black')
    for r in m.itertuples():
        ax.annotate(r.product_group, (r.lift_pct, r.ltv_ate), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.axhline(0, color='k', ls='--', lw=1); ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('F2 effect (Lift%)'); ax.set_ylabel('LTV effect (repeat revenue ATE)')
    ax.set_title(f'H&M: F2 vs LTV effect by first-purchase category\nSpearman(F2, LTV) = {rho_f2ltv:+.2f}')
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, 'hm_f2_vs_ltv.png'), dpi=150, bbox_inches='tight')

    print('\n==== SUMMARY ====', flush=True)
    print(f'cohort N={len(coh):,}  F2 baseline={baseF2:.3f}  LTV baseline={baseLTV:.4f}', flush=True)
    print(f'categories tested={len(f2)}  F2 sig={int(f2["sig5"].sum())}  LTV sig={int(ltv["sig5"].sum())}', flush=True)
    print(f'Spearman(price, F2 effect) = {rho_f2:+.2f}', flush=True)
    print(f'Spearman(F2 effect, LTV effect) = {rho_f2ltv:+.2f}', flush=True)
    print(f'outputs in {a.out_dir}/  (elapsed {time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
