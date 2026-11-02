"""Run host fault injection: python tests/test_fault_injection [--case NAME]."""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILESYSTEM_CASES = [
    'disable_fallback',
    'root_read_timeouts', 'root_commit_timeouts', 'root_other_errors', 'root_all_disabled',
    'root_queue_wait', 'root_concurrent_reports', 'root_write_timeouts',
    'short_write_retried',
    'drop_write_prefix', 'drop_write_middle', 'drop_write_tail',
    'write_enospc_after_prefix',
    'drop_read_prefix', 'drop_read_middle', 'drop_read_tail',
    'read_eio_after_prefix', 'corrupt_read',
    'missing_binary', 'missing_metadata', 'empty_binary', 'truncated_binary', 'stale_metadata',
    'fsync_eio', 'fsync_enospc', 'fsync_erofs', 'mkdir_eacces', 'lookup_eio', 'rename_eio',
    'rename_collision', 'rename_collision_bad_winner', 'rename_collision_uncommitted',
    'hang_read', 'hang_read_after_prefix', 'hang_write', 'hang_write_after_prefix',
    'hang_fsync', 'hang_commit', 'hang_rename', 'hang_mkdir', 'hang_lookup', 'all_roots_hang',
    'command_infinite', 'command_fifo', 'command_closed_output', 'command_active_output',
]
SITE_CASES = {
    'sites_cold': {'stat', 'mkdir', 'fopen_write', 'write', 'writev', 'fclose_write',
                   'open', 'fsync', 'close', 'openat', 'fdopendir', 'readdir', 'lstat',
                   'closedir', 'rename', 'fopen_read', 'read', 'fclose_read'},
    'sites_hit': {'stat', 'utimensat', 'fopen_read', 'read', 'fclose_read'},
    'sites_shared': {'fchmodat', 'fsync', 'rename', 'read'},
    'sites_cleanup': {'remove', 'readdir', 'closedir'},
}
CASES = list(SITE_CASES) + ['cache_regression', 'command_regression', *FILESYSTEM_CASES]


def stop_process_tree(process):
    # Each test has a private session; popen descendants inherit its process
    # group. Clean up the whole group when the watchdog fires or the test exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=CASES, help='run selected case(s), default: all')
    parser.add_argument('--list', action='store_true', help='list cases without compiling')
    parser.add_argument('--output-dir', type=Path, help='retain binaries, case data, logs and summary here')
    args = parser.parse_args()
    if args.list:
        print('\n'.join(CASES))
        return 0
    selected = args.case or CASES
    output = (args.output_dir or Path(tempfile.mkdtemp(prefix='deepjit-fault-injection-'))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data').mkdir(exist_ok=True)
    print(f'Artifacts: {output}', flush=True)

    def source_for(case):
        if case == 'cache_regression':
            return 'cache'
        if case == 'command_regression':
            return 'command'
        return 'filesystem'

    for source in sorted({source_for(case) for case in selected}):
        command = [os.environ.get('CXX', 'g++'), '-std=c++20', '-O2', '-pthread',
                   '-Wall', '-Wextra', '-Werror', '-I' + str(ROOT / 'include'),
                   str(HERE / f'{source}.cpp'), '-ldl', '-o', str(output / source)]
        if source in ('cache', 'filesystem'):
            command.append('-Wl,--wrap=fsync')
        print(f'[ BUILD ] {source}', flush=True)
        with (output / f'build-{source}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)

    def run(case, site=0, fault=None):
        name = f"{case}-{site}-{fault}" if site else case
        source = source_for(case)
        data = Path(tempfile.mkdtemp(prefix=name + '-', dir=output / 'data'))
        command = [str(output / source)]
        if source == 'filesystem':
            command += [case, str(data)]
            if site:
                command += [str(site), fault]
        else:
            command.append(str(data))
        log_path = output / f'{name}.log'
        timeout = 30
        print(f'[ RUN   ] {name}', flush=True)
        start = time.monotonic()
        timed_out = False
        with log_path.open('w') as log:
            env = dict(os.environ, DJ_FS_TIMEOUT_MS='100')
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process_tree(process)
                code = process.returncode
            finally:
                stop_process_tree(process)
        duration = time.monotonic() - start
        passed = code == 0 and not timed_out
        print(f'[ {"PASS" if passed else "FAIL":5} ] {name} ({duration:.3f}s)', flush=True)
        if not passed:
            print(f'  log: {log_path}\n' + '\n'.join(log_path.read_text().splitlines()[-8:]), flush=True)
        return dict(case=name, passed=passed, returncode=code, watchdog_timeout=timed_out,
                    seconds=duration, log=str(log_path))

    results = []
    for case in selected:
        baseline = run(case)
        results.append(baseline)
        if case in SITE_CASES and baseline['passed']:
            # Match the operation/path too: a changed execution order must
            # fail the replay instead of silently testing another site.
            def identity(line):
                fields = line.strip().split('\t')[1:]
                fields[-1] = re.sub(r'^.*?/primary(?=/|$)', 'primary', fields[-1])
                fields[-1] = re.sub(r'^primary/tmp/[^/]+', 'primary/tmp/{entry}', fields[-1])
                return fields
            sites = [identity(line) for line in Path(baseline['log']).read_text().splitlines()
                     if line.startswith('SITE\t')]
            missing = SITE_CASES[case] - {row[1] for row in sites}
            if missing:
                baseline['passed'] = False
                print(f"[ FAIL  ] {case}: unobserved operations {sorted(missing)}", flush=True)
            for expected in sites:
                for fault in ('error', 'hang'):
                    result = run(case, int(expected[0]), fault)
                    hits = [identity(line) for line in Path(result['log']).read_text().splitlines()
                            if line.startswith('HIT\t')]
                    result.update(site=expected, fault=fault)
                    if hits != [expected]:
                        result['passed'] = False
                        print(f"[ FAIL  ] site mismatch: expected {expected}, got {hits}", flush=True)
                    results.append(result)
    with (output / 'coverage.tsv').open('w') as report:
        report.write('flow\tsite\toperation\tpath\tfault\tresult\n')
        for result in results:
            if 'site' in result:
                flow = result['case'].split('-')[0]
                report.write('\t'.join([flow, *result['site'], result['fault'],
                                        'PASS' if result['passed'] else 'FAIL']) + '\n')
    passed = sum(result['passed'] for result in results)
    (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Fault injection: {passed}/{len(results)} passed. Results: {output / "summary.json"}', flush=True)
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
