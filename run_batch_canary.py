"""One explicit single-host experiment; no deployment changes or automatic retries."""
import argparse,json,subprocess,sys,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['batch1','batch2']);a=p.parse_args()
url='http://172.16.12.8:19600/v1'
for attempt in range(100):
 try:
  d=json.load(urllib.request.urlopen(url+'/models',timeout=3))
  if any(v['id']=='glm53-flash' for v in d['data']):break
 except Exception:pass
 time.sleep(3)
else:raise SystemExit('Canary not ready; no benchmark started')
base=['--accuracy-summary','/work/logs/gsm8k-2p2d-20261019/summary.json','--state-summary','/work/logs/state-healthy-20261019/summary.json']
def run(script,args):subprocess.run([sys.executable,'/work/'+script,*args],check=True)
if a.mode=='batch2':
 natural='/work/logs/natural-opt-batch2-20261019'
 run('benchmark_natural.py',base+['--url',url,'--limit','4','--concurrency','2','--output',natural])
 smoke=json.loads(Path(natural+'/summary.json').read_text());assert smoke['correct']==4
 state='/work/logs/state-opt-batch2-20261019'
 run('evaluate_state_reuse.py',['--url',url,'--lengths','1024,8192','--sequential-count','0','--concurrency','2','--output',state])
 base[-1]=state+'/summary.json'
run('benchmark_pd.py',base+['--url',url,'--topology','single-host-'+a.mode,'--input-lengths','1024','--output-tokens','64','--concurrencies','2','--requests','4','--repeats','2','--active-logical-devices','8','--installed-logical-devices','16','--allocated-hosts','1','--output','/work/logs/perf-opt-direct-'+a.mode+'-20261019'])
