"""
ReportResultsCollation.py

Reads ablation summary CSVs and writes benchmark-only data to a report-data
folder for direct use in a LaTeX report.

Only collected_benchmark and secondary_benchmark categories are extracted.
Transfer-learning runs (prefix: tf) and movement-summary files are excluded.

Outputs:
    <output_dir>/
        <prefix>_benchmark.csv   — benchmark metrics per run within the prefix group
        combined_benchmark.csv   — all groups merged, with an extra 'prefix' column

Usage:
    python ReportResultsCollation.py
    python ReportResultsCollation.py --summary_dir ./neural-network-models/summary-csv
    python ReportResultsCollation.py --output_dir ./report-data
"""

import argparse
import csv
import os


BENCHMARK_CATEGORIES = ['collected_benchmark', 'secondary_benchmark']

REPORT_METRICS = [
    'avg_r2',
    'avg_rmse',
    'Yaw_r2',
    'Pitch_r2',
    'Yaw_rmse',
    'Pitch_rmse',
    'mean_abs_latency_ms',
]


def _build_header():
    header = ['test_name']
    for category in BENCHMARK_CATEGORIES:
        for metric in REPORT_METRICS:
            header.append(f'{category}_{metric}')
            header.append(f'{category}_{metric}_iqr')
    return header


def _load_summary_csv(path):
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def _extract_row(raw_row, header):
    return {col: raw_row.get(col, '') for col in header}


def _write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col, '') for col in header])


def main():
    parser = argparse.ArgumentParser(
        description='Extract benchmark metrics from ablation summary CSVs for LaTeX report use.'
    )
    parser.add_argument(
        '--summary_dir',
        type=str,
        default='./neural-network-models/summary-csv',
        help='Directory containing *_summary.csv files (default: ./neural-network-models/summary-csv).',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./report-data',
        help='Output directory for report-ready CSVs (default: ./report-data).',
    )
    args = parser.parse_args()

    summary_dir = os.path.abspath(args.summary_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(summary_dir):
        print(f'[ERROR] Summary directory not found: {summary_dir}')
        return

    header = _build_header()
    combined_header = ['prefix'] + header
    combined_rows = []

    for filename in sorted(os.listdir(summary_dir)):
        if not filename.endswith('_summary.csv'):
            continue
        if '_movement_summary.csv' in filename:
            continue

        prefix = filename[: -len('_summary.csv')]

        if prefix == 'tf' or prefix.startswith('tf_'):
            print(f'[SKIP] Transfer-learning prefix: {filename}')
            continue

        path = os.path.join(summary_dir, filename)
        raw_rows = _load_summary_csv(path)

        if not raw_rows:
            print(f'[WARN] Empty file: {filename}')
            continue

        extracted = [_extract_row(r, header) for r in raw_rows]

        out_path = os.path.join(output_dir, f'{prefix}_benchmark.csv')
        _write_csv(out_path, header, extracted)
        print(f'[OK] Wrote {out_path}  ({len(extracted)} run(s))')

        for row in extracted:
            combined_rows.append({'prefix': prefix, **row})

    if combined_rows:
        combined_path = os.path.join(output_dir, 'combined_benchmark.csv')
        _write_csv(combined_path, combined_header, combined_rows)
        print(f'[OK] Wrote {combined_path}  ({len(combined_rows)} total run(s))')
    else:
        print('[WARN] No benchmark data found — check --summary_dir path.')


if __name__ == '__main__':
    main()
