"""Change only the two assigned Decode model instances; keep P/router/K8s unchanged."""
import argparse,base64,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['on','off']);p.add_argument('--batch-size',type=int,choices=[1,2],default=1);a=p.parse_args()
root=Path(__file__).resolve().parent;remote='/data/wio/xllm-glm53-flash'
def run(n,cmd):
 r=subprocess.run(['python3',str(root/'remote_run.py'),str(n),cmd],capture_output=True,text=True,timeout=120)
 print(n,r.returncode,r.stdout[-400:],r.stderr[-200:],flush=True);r.check_returncode()
# Refuse to stop engines while our test clients are still active.
code="""from pathlib import Path
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\\0') if v]
 except OSError:continue
 if len(a)>1 and a[1].startswith(('/work/benchmark_','/work/evaluate_','/work/run_best_')):active.append(a[1])
assert not active,repr(active)
print('NO_ACTIVE_TEST_CLIENTS')
"""
run(11,'docker exec xllm-flash-build-20261016 python3 -c '+shlex.quote(code))
archive='before-graph-'+a.mode+'-b'+str(a.batch_size)+'-'+str(time.time_ns())
for n in [10,11]:
 for name in ['launch_flash_baseline.sh','launch_flash_pd.sh','stop_flash_instance.py']:
  data=base64.b64encode((root/name).read_bytes()).decode()
  code=f'import base64;from pathlib import Path;Path({remote+"/"+name!r}).write_bytes(base64.b64decode({data!r}))'
  run(n,'python3 -c '+shlex.quote(code))
 c='xllm-flash-decode-healthy-20261019' if n==11 else 'xllm-flash-runtime-20261019'
 run(n,f'mkdir -p {remote}/logs/{archive}; cp -a {remote}/logs/decode {remote}/logs/{archive}/; docker exec {c} python3 /work/stop_flash_instance.py decode')
for n in [10,11]:
 c='xllm-flash-decode-healthy-20261019' if n==11 else 'xllm-flash-runtime-20261019'
 graph='true' if a.mode=='on' else 'false'
 run(n,f"docker exec -d {c} bash -c 'export XLLM_ENABLE_GRAPH={graph} XLLM_GRAPH_MAX_TOKENS={a.batch_size} XLLM_MAX_SEQS={a.batch_size}; bash /work/launch_flash_pd.sh {n} > /work/logs/graph-{a.mode}-launch.log 2>&1'")
print('DECODE_GRAPH_LAUNCHED',a.mode,'batch',a.batch_size,'WAIT_READY',flush=True)
