import subprocess,json
from pathlib import Path
root=Path('/workspace/experiments/forward-recovery-20260922/final')
jobs=[
 ('proof-eager',600,['python','scripts/check_bucketed_n.py']),
 ('proof-compiled',600,['python','scripts/check_bucketed_n.py','--torch-compile']),
 ('pytest-eager',1800,['python','-m','pytest','-q','tests/unit/test_fused.py','tests/unit/test_fused_contracts.py','tests/unit/test_fused_tma.py','-m','not compile']),
 ('pytest-compile',300,['python','-m','pytest','-q','tests/unit/test_fused.py','-m','compile']),
 ('large-reference',300,['python','scripts/validate_fused.py','--shapes','500,512,513,800,1024']),
 ('memcheck',300,['compute-sanitizer','--tool','memcheck','--error-exitcode','99','python','-m','pytest','-q','tests/unit/test_fused.py','-m','sanitizer']),
 ('initcheck',300,['compute-sanitizer','--tool','initcheck','--error-exitcode','99','python','-m','pytest','-q','tests/unit/test_fused.py','-m','sanitizer']),
 ('bench-workspace',600,['python','scripts/bench_fused.py','--baseline','full-workspace','--candidate','low-memory','--shapes','512,800,1024']),
 ('bench-final',600,['python','scripts/bench_fused.py','--shapes','500,512,513,640,768,800,1024']),
]
for name,timeout,cmd in jobs:
 rc=subprocess.call(['python','/workspace/experiments/fused-backward-20260921/run_logged.py','--log',str(root/(name+'.log')),'--timeout',str(timeout),'--',*cmd])
 print(json.dumps(dict(gate=name,returncode=rc)),flush=True)
 if rc:raise SystemExit(rc)
print(json.dumps(dict(all_passed=True)),flush=True)
