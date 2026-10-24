"""Regenerate Figures 2–4 using the original random seeds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import time
from run import ROOT, METHODS, REGIMES, NT, baseline, fit, cell, plots, np, pd

REPS = 1000
BASELINE_SEED = 5
POWER_SEED = 20260911


def normal_count(n, T):
    """One original ARMA, AR, or IID panel: alpha plus T+14 innovation pairs."""
    return n * (2 * T + 29)


def advance_normals(rng, count):
    """Consume the legacy Gaussian stream in bounded-memory chunks."""
    while count:
        chunk = min(count, 1_000_000)
        rng.standard_normal(chunk)
        count -= chunk


def baseline_cell(task):
    n, T, regime, state = task
    np.random.set_state(state)
    rho = [.6, 1 - T**(-.75), 1 - 1/T, 1., 1 + 1/T][REGIMES.index(regime)]
    values = []
    for _ in range(REPS):
        x, y = baseline.generate_data_ARMA(n, T, rho, .5, .4,
                                         np.array([[1, -.95], [-.95, 1]]), 0.)
        values.append(fit(baseline, x, y))
        # The old loop generated an AR panel and an IID panel next.
        # Their regressions are outside the main text. Reproduce their RNG
        # consumption without fitting or exporting those appendix experiments.
        advance_normals(np.random, 2 * normal_count(n, T))
    values = np.asarray(values)
    row = dict(N=n, T=T, rho_values=regime)
    for j, method in enumerate(METHODS):
        b, se = values[:, j, 0], values[:, j, 1]
        row.update({method+'-beta': b.mean(), method+'-se': se.mean(),
                    method+'-rmse': np.sqrt(np.mean(b*b)),
                    method+'-cp': np.mean(np.abs(b/se) <= 1.96)})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    out = ROOT / 'output/simulation/historical'
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    rng = np.random.RandomState(BASELINE_SEED)
    baseline_rows, power_rows = [], []
    # Original nesting: N, T, rho, replication; each repetition generates
    # ARMA -> AR -> IID. T=50 and rho=.2/.4 precede main-text cells too.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for n in [50, 100, 200]:
            for T in [50, 100, 200, 500]:
                for regime in ['0.2', '0.4', *REGIMES]:
                    if T != 50 and regime in REGIMES:
                        futures.append(pool.submit(baseline_cell,
                            (n, T, regime, rng.get_state())))
                    # Draw and discard the same normal sequence, never substitute
                    # MT19937 bit skips: the Gaussian sampler uses rejection.
                    advance_normals(rng, REPS * 3 * normal_count(n, T))
                print(f'Original seed-5 stream replayed through n={n}, T={T}; '
                      f'{time.monotonic()-start:.1f}s', flush=True)
        power_futures = [pool.submit(cell,
            (n, T, regime, REPS, POWER_SEED))
            for n, T in NT for regime in REGIMES]
        for future in as_completed(futures):
            row = future.result()
            baseline_rows.append(row)
        base = pd.DataFrame(baseline_rows).sort_values(['N', 'T', 'rho_values'])
        # The published baseline CSV used six decimal places. Use that same
        # serialization before plotting RMSE-derived standard deviations.
        base.to_csv(out / 'baseline.csv', index=False, float_format='%.6f')
        print('Baseline simulations complete.', flush=True)
        for i, future in enumerate(as_completed(power_futures)):
            power_rows.extend(future.result())
            print(f'Production power panels regenerated: {i+1}/20; '
                  f'{time.monotonic()-start:.1f}s', flush=True)
    power = pd.DataFrame(power_rows)
    base_index = base.set_index(['N', 'T', 'rho_values'])
    for i, row in power[power.beta == 0].iterrows():
        original_null = base_index.loc[(row.N, row['T'], row.rho_values)]
        for method in METHODS:
            power.loc[i, method+'-rr'] = 1 - original_null[method+'-cp']
    # Restore the exact final plotting order, without reading an output reference.
    power['_panel'] = [NT.index((n, T)) for n, T in zip(power.N, power['T'])]
    power['_regime'] = power.rho_values.map(REGIMES.index)
    power = power.sort_values(['_panel', '_regime', 'beta']).drop(columns=['_panel', '_regime'])
    power.to_csv(out / 'rejection_rates.csv', index=False)
    elapsed = time.monotonic() - start
    plots(out / 'baseline.csv', out)
    print(f'Simulations complete: {elapsed:.1f}s. Results: {out}', flush=True)

if __name__ == '__main__':
    main()
