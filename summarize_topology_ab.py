"""Combine the two matched small A/B runs; never equate this with peak performance."""
import csv
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parent
runs={m:root/('results/perf-ab-'+m+'-20261019') for m in ['pd','combined']}
cells={m:[json.loads(p.read_text()) for p in sorted(r.glob('cell-*.json'))] for m,r in runs.items()}
assert all(len(v)==8 for v in cells.values())
for m,r in runs.items():
 route=json.loads((r/'routing.json').read_text())
 assert route['routing_valid'] and route['all_four_used'] and route['measured_requests']==32
 assert all(c['summary']['successes']==c['summary']['requests']==4 for c in cells[m])
by_salt={m:{v['salt']:v for c in cs for v in c['requests']} for m,cs in cells.items()}
assert by_salt['pd'].keys()==by_salt['combined'].keys()
for salt,v in by_salt['pd'].items():
 w=by_salt['combined'][salt]
 assert v['tokenizer_prompt_tokens']==w['tokenizer_prompt_tokens']
 assert v['usage']['prompt_tokens']==w['usage']['prompt_tokens']
 assert v['usage']['completion_tokens']==w['usage']['completion_tokens']==128
rows=[]
for inp in [1024,8192]:
 for concurrency in [1,4]:
  r={'input_target':inp,'concurrency':concurrency}
  for m,cs in cells.items():
   matched=[c for c in cs if c['summary']['input_target']==inp and c['summary']['concurrency']==concurrency]
   reqs=[v for c in matched for v in c['requests']]
   duration=sum(c['summary']['duration'] for c in matched)
   r[m+'_output_tps']=sum(v['usage']['completion_tokens'] for v in reqs)/duration
   for metric in ['ttft','latency','tpot']:
    r[m+'_'+metric+'_p50']=statistics.median(v[metric] for v in reqs)
   r[m+'_repeat_tps']=[c['summary']['output_tps'] for c in matched]
  r['combined_output_tps_ratio']=r['combined_output_tps']/r['pd_output_tps']
  rows.append(r)
smoke=json.loads((root/'results/natural-ab-combined-20261019/summary.json').read_text())
assert smoke['successes']==smoke['natural_stops']==smoke['visible_answers']==smoke['correct']==8
report={'scope':'four hosts TP8 each; batch1; fixed128 reasoning-inclusive output; two repeats of four requests per cell; not peak/P99/full accuracy',
 'same_router':True,'matching_prompt_and_output_usage':True,'measured_requests_per_topology':32,'combined_smoke':smoke,
 'rows':rows,'combined_faster_in_all_cells':all(r['combined_output_tps_ratio']>1 for r in rows)}
out=root.parents[1]/'publication/inference-framework-notes'
(out/'topology-ab-summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'topology-ab-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:print(json.dumps(r))
