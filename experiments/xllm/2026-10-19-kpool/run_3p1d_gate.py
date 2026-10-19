"""Check exact 3P1D membership, then a small known-answer functional gate."""
import argparse,json,subprocess,sys,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--skip-long',action='store_true');a=p.parse_args()
root=Path('/work/logs');label='3p1d-'+a.profile+'-20261019';state={'phase':'readiness','label':label,'full_accuracy':False}
status=root/('diag-'+label+'-status.json')
def save():status.write_text(json.dumps(state,indent=2))
save()
try:
 for _ in range(120):
  try:
   for n in range(8,12):assert json.load(urllib.request.urlopen(f'http://172.16.12.{n}:19600/v1/models',timeout=2))['data']
   keys=subprocess.check_output(['/work/toolchains/etcd-v3.5.21-linux-arm64/etcdctl','--endpoints=http://172.16.12.11:12389','get','/flash-20261016/XLLM:','--prefix','--keys-only'],text=True,timeout=5)
   assert keys.count('XLLM:PREFILL:')==3 and keys.count('XLLM:DECODE:')==1 and 'XLLM:DEFAULT:' not in keys
   assert json.load(urllib.request.urlopen('http://172.16.12.11:28888/v1/models',timeout=2))['data']
   state['membership']=keys;break
  except (AssertionError,OSError,ValueError,subprocess.SubprocessError):time.sleep(3)
 else:raise RuntimeError('Expected 3P1D membership not ready; no requests sent')
 state['phase']='functional_gate';save()
 for length in ([1024,8192] if a.skip_long else [1024,8192,81920]):
  with (root/('diag-'+label+'-state-'+str(length)+'.log')).open('w') as f:
   result=subprocess.run([sys.executable,'/work/evaluate_state_reuse.py','--lengths',str(length),'--max-tokens','600','--concurrency','4','--sequential-count','0','--output',str(root/('state-'+label+'-'+str(length)))],stdout=f,stderr=subprocess.STDOUT)
  directory=root/('state-'+label+'-'+str(length))
  summary=json.loads((directory/'summary.json').read_text())
  state.setdefault('known_answer_results',{})[str(length)]=summary
  if length < 81920:
   result.check_returncode();state['last_passed_length']=length
  else:
   rows=[json.loads(p.read_text()) for p in directory.glob('concurrent-*.json')]
   assert len(rows)==4 and all(r.get('response',{}).get('usage',{}).get('completion_tokens',0)>0 for r in rows), 'Long requests had transport/execution errors'
   state['long_context_accuracy_passed']=summary['all_passed']
   state['performance_scope']='mechanical throughput, not model accuracy; short 1K/8K regression required, 81K retrieval recorded as observation because intrinsic model quality is not controlled'
  save()
 state['phase']='passed';state['gate_scope']='short 1K/8K regression only; no full accuracy claim' if a.skip_long else state.get('performance_scope','known-answer checks');save()
except Exception as error:state.update(phase='failed',error=repr(error));save();raise
