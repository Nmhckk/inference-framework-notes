"""Run the server-supplied AISBench wrapper with the user's exact workload."""
import argparse,json,os,subprocess,sys,time,shutil,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--profile',default='no-graph');p.add_argument('--performance-only',action='store_true');p.add_argument('--exclude-prefix-ids');a=p.parse_args()
logs=Path('/work/logs');root=logs/('perf-3p1d-aisbench-'+a.profile+'-20261019');assert not (root/'manifest.json').exists() and not (root/'wrapper.log').exists(), 'Refuse to overwrite a prior benchmark';root.mkdir(exist_ok=True);(root/'dataset').mkdir(exist_ok=True)
status=logs/('diag-3p1d-aisbench-'+a.profile+'-20261019-status.json');state={'phase':'waiting_for_gate','full_accuracy':False}
def save():status.write_text(json.dumps(state,indent=2))
save()
try:
 gate=logs/('diag-3p1d-'+a.profile+'-20261019-status.json')
 for _ in range(400):
  if a.performance_only:
   try:
    for n in [8,9,10,11]:assert json.load(urllib.request.urlopen(f'http://172.16.12.{n}:19600/v1/models',timeout=2))['data']
    assert json.load(urllib.request.urlopen('http://172.16.12.11:28888/v1/models',timeout=2))['data']
    break
   except (OSError,AssertionError,ValueError):time.sleep(3);continue
  if gate.exists():
   g=json.loads(gate.read_text())
   if g['phase']=='failed':raise RuntimeError('Candidate functional gate failed; AISBench was not started')
   if g['phase']=='passed':break
  time.sleep(3)
 else:raise RuntimeError('Functional gate incomplete; AISBench was not started')
 args=[sys.executable,'aisbench_test.py','--input_len','81920','--output_len','600','--data_num','48','--concurrency','16','--request_rate','0.93','--dataset_type','prefix_cache','--repeat_rate','0.93','--prefix_test','--dp','2']
 work=Path('/work/bench-3p1d-aisbench-'+a.profile+'-20261019')
 shutil.copytree('/work/bench-3p1d-aisbench-20261019',work)
 if a.exclude_prefix_ids:
  prefix_record=Path(a.exclude_prefix_ids);assert str(prefix_record).startswith('/work/bench-3p1d-aisbench-');shutil.copyfile(prefix_record,work/'picked_ids.txt')
 config=work/'config.py';config.write_text(config.read_text().replace('/work/logs/perf-3p1d-aisbench-20261019',str(root)))
 manifest={'command':args,'prefix_generation_exclusion_record':a.exclude_prefix_ids,'prefix_selection_scope':'exclude prior failed-run prefix IDs to avoid extra cache hits; workload parameters unchanged','source':'read-only copy from node22 /data/yxy/ais_bench/aisbench_auto_tools_prefix-main','profile':a.profile,'quality_validation':'NOT PASSED: short/long regression failures recorded separately' if a.performance_only else 'see referenced candidate gate','performance_only':a.performance_only,'target_input_tokens':81920,'target_output_tokens':600,'requests':48,'concurrency_cap':16,'request_rate':.93,'configured_prefix_ratio':.93,'warmup_prefix_copies':2,'actual_cache_hit_rate':'to be measured from xLLM; vLLM metric parser not applicable','active_logical_devices':32,'hosts':4,'topology':'3P1D','engine_graph':'verify runtime','MTP_speculative_tokens':'verify runtime','comparison_reference':{'source':'user screenshot','ttft_seconds':10.624,'tpot_ms':16.222,'output_tps':426.77,'hardware_and_tool_version':'not supplied'},'limits':['wrapper dp flag controls prefix prewarm copies, not server DP','wrapper npu_num defaults1: ignore its per-device figures; aggregate metrics unaffected','same script and generator settings; identical vLLM dataset and installed AISBench version not established']}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
 env=dict(os.environ,PYTHONPATH='/work/benchmark-tools/aisbench-3.1.0-owned',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0')
 state['phase']='running';save()
 with (root/'wrapper.log').open('w') as f:r=subprocess.run(args,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT)
 state.update(phase='client_finished_results_require_validation',exit_code=r.returncode);save()
 raise SystemExit(r.returncode)
except Exception as error:state.update(phase='failed',error=repr(error));save();raise
