"""ARLGOS: run a command, timestamp every output line into a log, sample the machine's RAM.

    python arlgos/run_logged.py logs/NAME.log -- python -u driver_unicorn.py

Each line the command prints (stdout and stderr, including the Ray workers'
lines that the driver forwards) goes to the console and to the log as
"<ISO time> <line>". arlgos/timing.py reads these times. Every 30 s the used
and total RAM go to logs/NAME.mem.csv. Exits with the command's exit code.
Works on Windows and Linux (WSL), so T4 can reuse it.
"""
import datetime
import os
import subprocess
import sys
import threading

from common import ram_mb


def now():
    return datetime.datetime.now().isoformat(timespec='milliseconds')


def sample_memory(path, stop, every=30):
    with open(path, 'a') as f:
        f.write('time,used_mb,total_mb\n')
        while True:
            used, total = ram_mb()
            f.write('{},{},{}\n'.format(now(), used, total))
            f.flush()
            if stop.wait(every):
                return


def main():
    log, command = sys.argv[1], sys.argv[sys.argv.index('--') + 1:]
    os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
    stop = threading.Event()
    sampler = threading.Thread(target=sample_memory, args=(os.path.splitext(log)[0] + '.mem.csv', stop), daemon=True)
    sampler.start()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', bufsize=1)
    with open(log, 'a', encoding='utf-8') as f:
        f.write('{} [run_logged] start: {}\n'.format(now(), ' '.join(command)))
        try:
            for line in proc.stdout:
                stamped = '{} {}'.format(now(), line.rstrip('\n'))
                print(stamped, flush=True)
                f.write(stamped + '\n')
                f.flush()
        except KeyboardInterrupt:
            proc.terminate()
        code = proc.wait()
        f.write('{} [run_logged] exit code {}\n'.format(now(), code))
    stop.set()
    sampler.join()
    sys.exit(code)


if __name__ == '__main__':
    main()
