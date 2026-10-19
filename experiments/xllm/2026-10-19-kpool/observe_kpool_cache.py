"""Read xLLM's native admission counters around the unmodified AISBench wrapper."""
from concurrent.futures import ThreadPoolExecutor
import json,re,time,urllib.request
from pathlib import Path
root=Path('/work/logs/perf-3p1d-aisbench-kpool-reuse-eager-20261019')
status=Path('/work/logs/diag-3p1d-aisbench-kpool-reuse-eager-20261019-status.json')
record={'scope':'P leader process admission counters; main-phase delta includes AISBench default one warmup request, not only48 measured requests','samples':{}}
def sample(label):
 def read(n):
  try:
   text=urllib.request.urlopen(f'http://172.16.12.{n}:19600/vars/prefix_cache*',timeout=5).read().decode()
   pairs=re.findall(r'id="value-(prefix_cache_(?:prompt|hit)_tokens_total)">([^<]+)',text)
   values={k:float(v) for k,v in pairs};assert len(values)==2
   return str(n),values
  except Exception as error:return str(n),{'error':repr(error)}
 with ThreadPoolExecutor(max_workers=3) as pool:rows=dict(pool.map(read,[8,9,10]))
 record['samples'][label]={'client_monotonic':time.perf_counter(),'P_counters':rows}
 root.mkdir(exist_ok=True);(root/'cache-counter-observation.json').write_text(json.dumps(record,indent=2))
sample('before_wrapper')
for _ in range(1200):
 text=(root/'wrapper.log').read_text(errors='replace') if (root/'wrapper.log').exists() else ''
 if '[开始] 全量数据集测试' in text and 'before_main_phase' not in record['samples']:sample('before_main_phase')
 if status.exists():
  state=json.loads(status.read_text())
  if state['phase'] in ['client_finished_results_require_validation','failed']:
   sample('after_wrapper');break
 time.sleep(2)
else:record['timeout']=True
if all(k in record['samples'] for k in ['before_main_phase','after_wrapper']):
 before=record['samples']['before_main_phase']['P_counters'];after=record['samples']['after_wrapper']['P_counters']
 if all('error' not in v for v in [*before.values(),*after.values()]):
  prompt=sum(after[n]['prefix_cache_prompt_tokens_total']-before[n]['prefix_cache_prompt_tokens_total'] for n in before)
  hit=sum(after[n]['prefix_cache_hit_tokens_total']-before[n]['prefix_cache_hit_tokens_total'] for n in before)
  record['main_phase_admission_counter_delta']={'prompt_tokens':prompt,'hit_tokens':hit,'token_hit_rate':hit/prompt if prompt else None}
(root/'cache-counter-observation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
