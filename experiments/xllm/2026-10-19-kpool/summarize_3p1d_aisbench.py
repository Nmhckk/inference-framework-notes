"""Validate the48-request AISBench report before publishing its three metrics."""
import argparse,csv,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--graph',action='store_true');p.add_argument('--mtp',type=int,default=1);a=p.parse_args()
root=Path(__file__).resolve().parent/'results'/a.run
candidates=[]
for path in root.glob('outputs/*/performances/vllm-api-stream-chat/gsm8k.json'):
 summary=json.loads(path.read_text())
 if summary.get('Total Requests',{}).get('total')==48:candidates.append((path,summary))
assert len(candidates)==1, 'No unique completed48-request report; do not substitute a prefix warmup report'
path,summary=candidates[0]
assert summary['Failed Requests']['total']==0 and summary['Success Requests']['total']==48, summary
assert summary['Total Generated Tokens']['total']==48*600, summary
latency={r['Performance Parameters']:r for r in csv.DictReader(path.with_suffix('.csv').open()) if r.get('Performance Parameters')}
rows=[json.loads(l) for l in path.with_name('gsm8k_details.jsonl').read_text().splitlines() if l.strip()]
assert len(rows)==48 and all(r['success'] and r['output_tokens']==600 for r in rows)
def numeric(value):return float(re.match(r'[-+0-9.eE]+',str(value)).group())
duration=numeric(summary['Benchmark Duration']['total'])/1000
output_tps=numeric(summary['Output Token Throughput']['total'])
assert abs(output_tps-28800/duration)<.001, (output_tps,duration)
result={'run':a.run,'scope':'AISBench performance pressure data only; quality not validated','requests':48,'successes':48,'failed':0,'output_tokens':28800,'actual_input_lengths':dict(sorted((str(n),sum(r['input_tokens']==n for r in rows)) for n in {r['input_tokens'] for r in rows})),'actual_output_lengths':{'600':48},'ttft_mean_seconds':numeric(latency['TTFT']['Average'])/1000,'ttft_median_seconds':numeric(latency['TTFT']['Median'])/1000,'ttft_p90_seconds':numeric(latency['TTFT']['P90'])/1000,'tpot_mean_ms':numeric(latency['TPOT']['Average']),'tpot_median_ms':numeric(latency['TPOT']['Median']),'tpot_p90_ms':numeric(latency['TPOT']['P90']),'cluster_output_tps':output_tps,'benchmark_duration_seconds':duration,'max_client_concurrency':summary['Max Concurrency']['total'],'observed_average_concurrency':summary['Concurrency']['total'],'resources':{'hosts':4,'active_logical_devices':32,'topology':'3P1D','TP_per_instance':8,'D_batch_limit':8,'D_memory_utilization':.75,'graph':a.graph,'MTP_tokens':a.mtp,'prefix_cache':True,'schedule_overlap':True},'source_report':str(path.relative_to(root)),'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'dataset').glob('*.jsonl')},'native_cache_observation':json.loads((root/'cache-counter-observation.json').read_text()),'native_mtp_observation':json.loads((root/'mtp-counter-observation.json').read_text()),'AISBench_aggregates':summary,'AISBench_latency_rows':latency}
(root/'validated-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['native_cache_observation','native_mtp_observation','AISBench_aggregates','AISBench_latency_rows']},ensure_ascii=False,indent=2))
