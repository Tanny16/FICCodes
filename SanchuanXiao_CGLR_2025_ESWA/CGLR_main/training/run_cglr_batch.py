import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = Path(__file__).resolve().parent


DATASETS = [
    {'dataset': 'CAS_B', 'family': 'CAS-B', 'data_type': 'raw', 'path': 'data/CAS_B.csv', 'code_num': 305},
    {'dataset': 'CAS_S', 'family': 'CAS-S', 'data_type': 'raw', 'path': 'data/CAS_S.csv', 'code_num': 771},
    {'dataset': 'CAS_T', 'family': 'CAS-T', 'data_type': 'raw', 'path': 'data/CAS_T.csv', 'code_num': 376},
]


EPOCH_PATTERN = re.compile(
    r'Epoch:\s*(?P<epoch>\d+).*?R2_val:\s*(?P<r2_val>[-+0-9.eE]+).*?'
    r'R2_test:\s*(?P<r2_test>[-+0-9.eE]+).*?'
    r'Rank_IC_test:\s*(?P<rank_ic_test>[-+0-9.eE]+).*?'
    r'Rank_ICIR_test:\s*(?P<rank_icir_test>[-+0-9.eE]+)'
)


SUMMARY_FIELDS = [
    'dataset', 'family', 'data_type', 'status', 'data_file', 'code_num',
    'time_count', 'feature_count', 'epochs_completed', 'best_test_r2',
    'best_test_epoch', 'val_r2_at_best_test', 'rank_ic_test',
    'rank_icir_test', 'final_test_r2', 'return_code', 'run_dir', 'error'
]


def build_command(item, run_dir):
    data_file_arg = str(Path('..') / item['path'])
    run_dir_relative = run_dir.resolve().relative_to(PROJECT_ROOT.resolve())
    run_dir_arg = str(Path('..') / run_dir_relative)
    return [
        sys.executable,
        'regression.py',
        '--data_file', data_file_arg,
        '--code_num', str(item['code_num']),
        '--run_name', item['dataset'],
        '--data_type', item['data_type'],
        '--run_dir', run_dir_arg,
        '--seed', '2023',
        '--patience', '20',
        '--min_epochs', '30',
        '--accumulation_steps', '1',
        '-length', '15',
        '-feature', '9',
        '-n_class', '1',
        '-epoch', '300',
        '-batch_size', '32',
        '--rnn_unit', '50',
        '-d_model', '16',
        '--hidden', '100',
        '-dropout', '0.20',
        '--weight-constraint', '5e-4',
        '--lr', '5e-4',
        '--use_hidden_rel', '1',
        '--max_step', '10',
        '--num_path', '10',
        '--top_k', '6',
    ]


def append_summary(summary_path, record):
    is_new = not summary_path.exists()
    with summary_path.open('a', newline='', encoding='utf-8-sig') as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({field: record.get(field, '') for field in SUMMARY_FIELDS})


def stream_run(item, run_dir):
    command = build_command(item, run_dir)
    console_path = run_dir / 'console.log'
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    print(f"[START] {item['family']} | {item['dataset']} | {item['data_type']}", flush=True)
    print(f"[PATH] {item['path']}", flush=True)
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(TRAINING_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=environment,
    )
    max_seen = float('-inf')
    captured_tail = []
    with console_path.open('w', encoding='utf-8') as console_file:
        for line in process.stdout:
            console_file.write(line)
            console_file.flush()
            captured_tail.append(line.rstrip())
            if len(captured_tail) > 30:
                captured_tail.pop(0)
            match = EPOCH_PATTERN.search(line)
            if match:
                epoch = int(match.group('epoch'))
                r2_val = float(match.group('r2_val'))
                r2_test = float(match.group('r2_test'))
                rank_ic_test = float(match.group('rank_ic_test'))
                rank_icir_test = float(match.group('rank_icir_test'))
                new_best = r2_test > max_seen
                if new_best:
                    max_seen = r2_test
                if epoch == 1 or epoch % 5 == 0 or new_best:
                    print(
                        f"[PROGRESS] {item['dataset']} epoch={epoch} "
                        f"R2_val={r2_val:.4f} R2_test={r2_test:.4f} "
                        f"Rank_IC_test={rank_ic_test:.4f} "
                        f"Rank_ICIR_test={rank_icir_test:.4f}"
                        + (' [NEW_BEST_TEST]' if new_best else ''),
                        flush=True,
                    )
    return_code = process.wait()
    duration = time.time() - started
    return return_code, duration, captured_tail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', default=str(PROJECT_ROOT / 'batch_runs'))
    args = parser.parse_args()

    batch_root = Path(args.output_root).resolve()
    batch_dir = batch_root / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    summary_path = batch_dir / 'summary.csv'
    manifest_path = batch_dir / 'manifest.csv'
    with manifest_path.open('w', newline='', encoding='utf-8-sig') as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=['dataset', 'family', 'data_type', 'path', 'code_num'])
        writer.writeheader()
        writer.writerows(DATASETS)

    print(f'[BATCH] output_dir={batch_dir}', flush=True)
    print(f'[BATCH] dataset_count={len(DATASETS)}', flush=True)
    completed = 0

    for index, item in enumerate(DATASETS, start=1):
        run_dir = batch_dir / f"{index:02d}_{item['dataset']}"
        data_path = (PROJECT_ROOT / item['path']).resolve()
        if not data_path.is_file():
            record = {
                'dataset': item['dataset'],
                'family': item['family'],
                'data_type': item['data_type'],
                'status': 'failed_data_validation',
                'data_file': str(data_path),
                'code_num': item['code_num'],
                'return_code': '',
                'run_dir': str(run_dir),
                'error': 'data file not found',
            }
            append_summary(summary_path, record)
            print(f"[SKIP] missing data: {data_path}", flush=True)
            continue

        return_code, duration, captured_tail = stream_run(item, run_dir)
        summary_file = run_dir / 'summary.json'
        if return_code == 0 and summary_file.is_file():
            import json
            with summary_file.open('r', encoding='utf-8') as summary_json:
                result = json.load(summary_json)
            record = {
                'dataset': item['dataset'],
                'family': item['family'],
                'data_type': item['data_type'],
                'status': 'completed',
                'data_file': result.get('data_file', str(data_path)),
                'code_num': result.get('code_num', item['code_num']),
                'time_count': result.get('time_count', ''),
                'feature_count': result.get('feature_count', ''),
                'epochs_completed': result.get('epochs_completed', ''),
                'best_test_r2': result.get('best_test_r2', ''),
                'best_test_epoch': result.get('best_test_epoch', ''),
                'val_r2_at_best_test': result.get('val_r2_at_best_test', ''),
                'rank_ic_test': result.get('best_test_rank_ic', ''),
                'rank_icir_test': result.get('best_test_rank_icir', ''),
                'final_test_r2': result.get('final_test_r2', ''),
                'return_code': return_code,
                'run_dir': str(run_dir),
                'error': '',
            }
            append_summary(summary_path, record)
            completed += 1
            print(
                f"[DONE] {item['dataset']} best_test_R2={record['best_test_r2']} "
                f"epoch={record['best_test_epoch']} final_test_R2={record['final_test_r2']} "
                f"elapsed={duration / 60:.1f}min",
                flush=True,
            )
            continue

        error_text = '\n'.join(captured_tail[-10:])
        validation_error = any('DATA_VALIDATION_ERROR' in line for line in captured_tail)
        status = 'failed_data_validation' if validation_error else 'failed_runtime'
        record = {
            'dataset': item['dataset'],
            'family': item['family'],
            'data_type': item['data_type'],
            'status': status,
            'data_file': str(data_path),
            'code_num': item['code_num'],
            'return_code': return_code,
            'run_dir': str(run_dir),
            'error': error_text,
        }
        append_summary(summary_path, record)
        print(f"[FAIL] {item['dataset']} return_code={return_code}; see {run_dir / 'console.log'}", flush=True)
        if not validation_error:
            print('[STOP] runtime/environment error; later datasets were not started.', flush=True)
            break

    print(f'[BATCH_FINISHED] completed={completed}/{len(DATASETS)}', flush=True)
    print(f'[SUMMARY] {summary_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
