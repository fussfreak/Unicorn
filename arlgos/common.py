"""ARLGOS: shared helpers for the scripts in arlgos/ (machine and version info for config.json files)."""
import os
import platform
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpu_name():
    try:
        if os.name == 'nt':
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
            return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def ram_mb():
    """(used, total) physical memory in MB, for the whole machine."""
    if os.name == 'nt':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return (status.ullTotalPhys - status.ullAvailPhys) // 2 ** 20, status.ullTotalPhys // 2 ** 20
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            key, value = line.split(':', 1)
            info[key] = int(value.split()[0])
    return (info['MemTotal'] - info['MemAvailable']) // 1024, info['MemTotal'] // 1024


def env_sumo_home():
    """SUMO_HOME of the eclipse-sumo package installed in this env (the launchers export it)."""
    import sumo
    return sumo.SUMO_HOME


def use_env_sumo():
    """Point SUMO_HOME at this env's eclipse-sumo, as arlgos/*.ps1 do, so checkBinary finds it."""
    os.environ['SUMO_HOME'] = env_sumo_home()


def sumo_binary_version():
    """Version of the sumo binary that sumolib.checkBinary picks (via SUMO_HOME), e.g. '1.28.0'."""
    from sumolib import checkBinary
    binary = checkBinary('sumo')
    out = subprocess.run([binary, '--version'], capture_output=True, text=True).stdout
    match = re.search(r'(\d+\.\d+\.\d+)', out)
    return (match.group(1) if match else 'unknown'), binary


def git_state():
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return commit + ('+uncommitted' if dirty else '')
    except OSError:
        return 'unknown'


def machine_info():
    import numpy
    import torch
    try:
        import ray
        ray_version = ray.__version__
    except ImportError:
        ray_version = None
    import traci
    sumo_version, sumo_path = sumo_binary_version()
    return {'host': platform.node(), 'os': platform.platform(), 'cpu': cpu_name(),
            'logical_cpus': os.cpu_count(), 'ram_mb': ram_mb()[1],
            'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': numpy.__version__,
            'ray': ray_version, 'sumo': sumo_version, 'sumo_binary': sumo_path,
            'sumo_home': os.environ.get('SUMO_HOME'),
            'traci_api': traci.constants.TRACI_VERSION, 'fork_commit': git_state()}
