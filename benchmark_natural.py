"""Supplement: fixed-prefix GSM8K test questions, natural stopping, visible answer timing.

Run after the fixed-length matrix, without overlapping other test traffic.
This small sample is not a stable P99 or a new full accuracy benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import urllib.request
from evaluate_gsm8k import extract


def percentile(values,q):
    if not values:return None
    v=sorted(values);p=(len(v)-1)*q;i=int(p);j=min(i+1,len(v)-1)
    return v[i]+(v[j]-v[i])*(p-i)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://172.16.12.11:28888/v1')
    p.add_argument('--data',default='/work/gsm8k-test.jsonl')
    p.add_argument('--train',default='/work/gsm8k-train4.jsonl')
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--accuracy-summary',required=True)
    p.add_argument('--state-summary',required=True)
    a=p.parse_args()
    assert json.loads(Path(a.accuracy_summary).read_text())['regression_gate'] is True
    assert json.loads(Path(a.state_summary).read_text())['all_passed'] is True
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    data=[json.loads(x) for x in Path(a.data).read_text().splitlines()][:a.limit]
    assert 2 <= a.limit <= 1319 and a.concurrency > 0
    train=[json.loads(x) for x in Path(a.train).read_text().splitlines()][:4]
    examples='\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in train)
    manifest=dict(vars(a),indices=list(range(len(data))),concurrency=a.concurrency,measured_requests=len(data),warmup_requests=2,
                  few_shot=4,max_tokens=2048,temperature=1.0,top_p=.95,seed=42,ignore_eos=False,
                  ttft='first non-empty reasoning/content delta',ttfa_nonblank='first non-whitespace content delta',
                  timer='client time.perf_counter',scope='supplementary fixed-prefix natural-stop answer latency, one run')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def probe(spec):
        index,warmup=spec;item=data[index]
        prompt=('Solve the math word problem. Show your reasoning concisely, then end your final answer with '
                '#### followed by the numeric answer.\n\n'+examples+'\n\nQuestion: '+item['question']+'\nAnswer:')
        body={'model':'glm53-flash','messages':[{'role':'user','content':prompt}],
              'max_tokens':2048,'temperature':1.0,'top_p':.95,'seed':42,'stream':True,
              'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':True}}
        row={'index':index,'question':item['question'],'expected':extract(item['answer']),'warmup':warmup}
        start=time.perf_counter();first=None;answer_first=None;content=[];reasoning=[];events=[];usage=None;finish=None;done=False
        try:
            req=urllib.request.Request(a.url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=1200) as resp:
                for raw in resp:
                    line=raw.decode().strip()
                    if not line.startswith('data:'):continue
                    data_line=line[5:].strip();now=time.perf_counter()
                    if data_line=='[DONE]':done=True;break
                    chunk=json.loads(data_line)
                    if chunk.get('error'):raise RuntimeError(str(chunk['error']))
                    row.setdefault('request_id',chunk.get('id'))
                    if chunk.get('usage'):usage=chunk['usage']
                    for choice in chunk.get('choices',[]):
                        if choice.get('finish_reason'):finish=choice['finish_reason']
                        delta=choice.get('delta',{});answer=delta.get('content') or '';think=delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if answer or think:
                            if first is None:first=now
                            events.append({'seconds':now-start,'content':answer,'reasoning':think})
                        if answer.strip() and answer_first is None:answer_first=now
                        content.append(answer);reasoning.append(think)
            end=time.perf_counter()
            if not done or first is None or not usage:raise RuntimeError('Missing DONE, nonempty output or usage')
            predicted=extract(''.join(content));n=usage['completion_tokens']
            row.update(ok=True,predicted=predicted,correct=predicted==row['expected'],finish_reason=finish,
                       latency=end-start,ttft=first-start,ttfa_nonblank=None if answer_first is None else answer_first-start,
                       tpot=None if n<2 else (end-first)/(n-1),usage=usage,
                       content=''.join(content),reasoning=''.join(reasoning))
        except Exception as error:row.update(ok=False,correct=False,error=repr(error),latency=time.perf_counter()-start)
        row.update(done_received=done,finish_observed=finish,usage_observed=usage,partial_content=''.join(content),partial_reasoning=''.join(reasoning))
        row['events']=events
        (out/f'{"warmup" if warmup else "item"}-{index:02d}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('content','reasoning','question','events')},ensure_ascii=False),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:warm=list(pool.map(probe,[(i,True) for i in range(2)]))
    if any(not r['ok'] for r in warm):raise SystemExit('Natural-stop warmup failed')
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:rows=list(pool.map(probe,[(i,False) for i in range(len(data))]))
    duration=time.perf_counter()-start;good=[r for r in rows if r['ok']]
    visible=[r['ttfa_nonblank'] for r in good if r['ttfa_nonblank'] is not None]
    summary={'requests':len(data),'concurrency':a.concurrency,'repeats':1,'successes':len(good),'correct':sum(r['correct'] for r in rows),
             'natural_stops':sum(r.get('finish_reason')=='stop' for r in rows),'visible_answers':len(visible),
             'duration':duration,'output_tps':sum(r['usage']['completion_tokens'] for r in good)/duration,
             'ttft_p50':percentile([r['ttft'] for r in good],.5),'ttfa_nonblank_p50':percentile(visible,.5),
             'ttfa_nonblank_p95':percentile(visible,.95),'latency_p50':percentile([r['latency'] for r in rows],.5),
             'latency_p95':percentile([r['latency'] for r in rows],.95),
             'scope':'one fixed-prefix natural-stop GSM8K supplement; not comparable to fixed128 benchmark throughput'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return 0 if len(good)==len(data) and summary['natural_stops']==len(data) and summary['visible_answers']==len(data) else 2

if __name__=='__main__':raise SystemExit(main())
