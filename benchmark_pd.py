"""Closed-loop SSE benchmark. Run only after the accuracy gate passes.

Uses client monotonic clocks. Usage is mandatory, all P+D resources count.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import time
import urllib.request


def quantile(values, q):
    if not values:
        return None
    data=sorted(values)
    position=(len(data)-1)*q
    lo,hi=math.floor(position),math.ceil(position)
    return data[lo]+(data[hi]-data[lo])*(position-lo)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://172.16.12.11:28888/v1')
    p.add_argument('--accuracy-summary',required=True)
    p.add_argument('--state-summary',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='/models/glm53-flash')
    p.add_argument('--requests',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--concurrencies',default='1,2,4')
    a=p.parse_args()
    gate=json.loads(Path(a.accuracy_summary).read_text())
    if gate.get('regression_gate') is not True:
        raise SystemExit('Accuracy regression gate is not passed; performance not started')
    state_gate=json.loads(Path(a.state_summary).read_text())
    if state_gate.get('all_passed') is not True:
        raise SystemExit('State/long-context regression gate is not passed')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    manifest=dict(vars(a),harness='custom urllib SSE, closed-loop',active_logical_devices=32,
                  installed_logical_devices=64,hosts=4,scope='conservative batch1 baseline; not tuned peak',
                  ttft='first non-empty reasoning or answer delta',
                  ttfa='first non-empty visible answer delta, nullable',
                  tpot='(end-to-end latency - TTFT)/(usage.completion_tokens - 1)',
                  time_source='one client time.perf_counter; not server wall clocks',
                  tail_limit='small sample; no stable P99 claim')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def make_messages(target,salt):
        # Keep prompts ordinary readable text; measure full chat-template tokens.
        prefix=f'Benchmark request {salt}. Continue the following numbered explanatory notes in detail.\n'
        filler='The system processes data and produces results. Each stage has inputs, outputs, and timing. '
        n=max(1,target//18)
        for _ in range(8):
            text=prefix+filler*n
            messages=[{'role':'user','content':text}]
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            length=len(tokenizer.encode(rendered,add_special_tokens=False))
            if abs(length-target)<=18:break
            n=max(1,n+round((target-length)/18))
        return messages,length
    def probe(spec):
        target,output_tokens,salt=spec
        messages,expected_prompt=make_messages(target,salt)
        body={'model':'glm53-flash','messages':messages,'max_tokens':output_tokens,
              'temperature':0,'stream':True,'stream_options':{'include_usage':True},
              'ignore_eos':True,'chat_template_kwargs':{'enable_thinking':True}}
        start=time.perf_counter();first=None;answer_first=None;last_chunk=None;events=[];usage=None;done=False;request_id=None
        result={'target_input_tokens':target,'target_output_tokens':output_tokens,'tokenizer_prompt_tokens':expected_prompt,'salt':salt}
        try:
            request=urllib.request.Request(a.url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=1200) as response:
                for raw in response:
                    line=raw.decode().strip()
                    if not line.startswith('data:'):continue
                    now=time.perf_counter();data=line[5:].strip()
                    if data=='[DONE]':done=True;break
                    chunk=json.loads(data);request_id=request_id or chunk.get('id')
                    if chunk.get('usage'):usage=chunk['usage']
                    text='';answer=''
                    for choice in chunk.get('choices',[]):
                        delta=choice.get('delta',{})
                        answer+=delta.get('content') or ''
                        text+=(delta.get('content') or '')+(delta.get('reasoning_content') or delta.get('reasoning') or '')
                    if text:
                        if first is None:first=now
                        events.append({'t':now-start,'chars':len(text),'gap':None if last_chunk is None else now-last_chunk})
                        last_chunk=now
                    if answer and answer_first is None:answer_first=now
            end=time.perf_counter()
            if not done or first is None or not usage:raise RuntimeError('Missing DONE, nonempty output or usage')
            completion=int(usage['completion_tokens'])
            if completion!=output_tokens:raise RuntimeError(f'Output length {completion} != fixed target {output_tokens}')
            result.update(ok=True,latency=end-start,ttft=first-start,ttfa=None if answer_first is None else answer_first-start,
                          tpot=None if completion<2 else (end-first)/(completion-1),usage=usage,
                          max_chunk_gap=max([e['gap'] for e in events if e['gap'] is not None] or [0]))
        except Exception as error:
            result.update(ok=False,error=repr(error),latency=time.perf_counter()-start,usage=usage)
        result.update(request_id=request_id,events=events)
        return result
    matrix=[]
    for repeat in range(a.repeats):
        for input_tokens,output_tokens in [(1024,128),(8192,128)]:
            for concurrency in map(int,a.concurrencies.split(',')):
                # Warmup each cell outside timed measurements.
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    warm=list(pool.map(probe,[(input_tokens,output_tokens,f'warm-{repeat}-{input_tokens}-{concurrency}-{i}') for i in range(concurrency)]))
                if any(not x['ok'] for x in warm):
                    (out/'warmup-failure.json').write_text(json.dumps(warm,indent=2));raise SystemExit('Warmup failed')
                start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows=list(pool.map(probe,[(input_tokens,output_tokens,f'{repeat}-{input_tokens}-{concurrency}-{i}') for i in range(a.requests)]))
                duration=time.perf_counter()-start
                successes=[r for r in rows if r['ok']]
                cell={'repeat':repeat,'input_target':input_tokens,'output_target':output_tokens,'concurrency':concurrency,
                      'requests':len(rows),'successes':len(successes),'duration':duration,
                      'output_tps':sum(r['usage']['completion_tokens'] for r in successes)/duration,
                      'total_tps':sum(r['usage']['completion_tokens']+r['usage']['prompt_tokens'] for r in successes)/duration,
                      'output_tps_per_active_logical_device':sum(r['usage']['completion_tokens'] for r in successes)/duration/32,
                      'output_tps_per_allocated_host':sum(r['usage']['completion_tokens'] for r in successes)/duration/4,
                      'latency_p50':quantile([r['latency'] for r in successes],.5),'latency_p95':quantile([r['latency'] for r in successes],.95),
                      'ttft_p50':quantile([r['ttft'] for r in successes],.5),'ttft_p95':quantile([r['ttft'] for r in successes],.95),
                      'tpot_p50':quantile([r['tpot'] for r in successes if r['tpot'] is not None],.5),
                      'tpot_p95':quantile([r['tpot'] for r in successes if r['tpot'] is not None],.95)}
                matrix.append(cell)
                (out/f'cell-{repeat}-{input_tokens}-{concurrency}.json').write_text(json.dumps({'summary':cell,'requests':rows,'warmup':warm},indent=2))
                (out/'summary.json').write_text(json.dumps(matrix,indent=2))
                print(json.dumps(cell),flush=True)
                if len(successes)!=len(rows):raise SystemExit('Measured requests failed; preserve results and investigate')
    return 0

if __name__=='__main__':raise SystemExit(main())
