"""Read xLLM's native admission counters around the unmodified AISBench wrapper."""
from concurrent.futures import ThreadPoolExecutor
import json,re,time,urllib.request
from pathlib import Path
root=Path('/work/logs/perf-3p1d-aisbench-kpool-reuse-eager-20261019')
status=Path('/work/logs/diag-3p1d-aisbench-kpool-reuse-eager-20261019-status.json')
record={'scope':'Decode native MTP counters; main phase includes AISBench default one warmup request','samples':{}}
def sample(label):
 def read(n):
  try:
   text=urllib.request.urlopen(f'http://172.16.12.{n}:19600/vars/speculative*',timeout=5).read().decode()
   pairs=re.findall(r'id="value-(speculative_num_(?:drafts|accepted_tokens|draft_tokens|committed_tokens)_total)">([^<]+)',text)
   values={k:float(v) for k,v in pairs};assert len(values)==4
   return str(n),values
  except Exception as error:return str(n),{'error':repr(error)}
 with ThreadPoolExecutor(max_workers=1) as pool:rows=dict(pool.map(read,[11]))
 record['samples'][label]={'client_monotonic':time.perf_counter(),'D_counters':rows}
 root.mkdir(exist_ok=True);(root/'mtp-counter-observation.json').write_text(json.dumps(record,indent=2))
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
 before=record['samples']['before_main_phase']['D_counters'];after=record['samples']['after_wrapper']['D_counters']
 if all('error' not in v for v in [*before.values(),*after.values()]):
  delta={k:after['11'][k]-before['11'][k] for k in before['11']}
  record['main_phase_delta']=delta
(root/'mtp-counter-observation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
