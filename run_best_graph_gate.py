"""Single ordered Decode-graph trial: readiness, correctness, then a matched screen."""
import json,subprocess,sys,time,urllib.request
from pathlib import Path
log=Path('/work/logs');status=log/'perf-best-graph-status-20261019.json'
state={'phase':'readiness','full_accuracy':False,'automatic_retry':False}
def save():status.write_text(json.dumps(state,indent=2))
def run(phase,script,args):
 state['phase']=phase;save()
 with (log/('perf-best-graph-'+phase+'-20261019.log')).open('w') as f:
  subprocess.run([sys.executable,'/work/'+script,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
for _ in range(100):
 try:
  for n in range(8,12):assert json.load(urllib.request.urlopen(f'http://172.16.12.{n}:19600/v1/models',timeout=3))['data']
  keys=subprocess.check_output(['/work/toolchains/etcd-v3.5.21-linux-arm64/etcdctl','--endpoints=http://172.16.12.11:12389','get','/flash-20261016/XLLM:','--prefix','--keys-only'],text=True,timeout=5)
  assert keys.count('XLLM:PREFILL:')==2 and keys.count('XLLM:DECODE:')==2 and 'XLLM:DEFAULT:' not in keys
  break
 except Exception:time.sleep(3)
else:raise SystemExit('Graph topology not ready; test not started')
try:
 run('state','evaluate_state_reuse.py',['--lengths','1024,8192','--sequential-count','0','--concurrency','2','--output','/work/logs/state-best-graph-20261019'])
 base=['--accuracy-summary','/work/logs/gsm8k-2p2d-20261019/summary.json','--state-summary','/work/logs/state-best-graph-20261019/summary.json']
 run('natural','benchmark_natural.py',base+['--limit','4','--concurrency','2','--output','/work/logs/natural-best-graph-20261019'])
 natural=json.loads((log/'natural-best-graph-20261019/summary.json').read_text());assert natural['correct']==natural['successes']==natural['natural_stops']==4
 gate={'regression_gate':True,'scope':'graph-specific state8 plus natural-stop math4; not full model accuracy','state_summary':'state-best-graph-20261019/summary.json','natural_summary':natural}
 (log/'perf-best-graph-gate-20261019.json').write_text(json.dumps(gate,indent=2))
 run('screen','benchmark_pd.py',['--accuracy-summary','/work/logs/perf-best-graph-gate-20261019.json','--state-summary','/work/logs/state-best-graph-20261019/summary.json','--topology','2p2d-decode-graph-screen','--input-lengths','1024','--output-tokens','128','--concurrencies','1,2,4','--requests','4','--repeats','1','--output','/work/logs/perf-best-graph-screen-20261019'])
 state['phase']='complete';save()
except Exception as error:
 state.update(phase='failed',error=repr(error));save();raise
