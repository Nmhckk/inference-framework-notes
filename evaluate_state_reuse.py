"""Known-answer long-context and interleaved-request state regression."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://172.16.12.11:28888/v1')
    p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='/models/glm53-flash')
    a=p.parse_args()
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for target in (1024,8192,16384):
        for variant in range(4):
            name=f'length{target}-variant{variant}'
            key=f'KEY_{target}_{variant}';value=f'VALUE_{target+173+variant*391}'
            ordinary='普通背景记录：系统接收输入，执行计算，并保存结果。此行没有目标键。\n'
            record=f'唯一目标记录：{key}={value}\n'
            question=f'从以下资料查找键 {key} 对应的完整值。最终回答只输出该值，不要解释。\n'
            n=max(1,target//23)
            for _ in range(8):
                cut=(n*variant)//3
                text=question+ordinary*cut+record+ordinary*(n-cut)+'\n请按资料回答。'
                messages=[{'role':'user','content':text}]
                tokens=len(t.encode(t.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
                if abs(tokens-target)<30:break
                n=max(1,n+round((target-tokens)/23))
            cases.append({'name':name,'messages':messages,'expected':value,'prompt_tokens_client':tokens})
    (out/'manifest.json').write_text(json.dumps({'cases':[{k:v for k,v in c.items() if k!='messages'} for c in cases],
        'max_tokens':1024,'temperature':0,'phases':['first4 sequential','all12 concurrency4'],
        'pass_rule':'all16 exact final-answer matches; no truncation or transport errors'},indent=2))
    def probe(spec):
        phase,case=spec
        body={'model':'glm53-flash','messages':case['messages'],'temperature':0,'max_tokens':1024,'stream':False,
              'chat_template_kwargs':{'enable_thinking':True}}
        row={k:v for k,v in case.items() if k!='messages'};row['phase']=phase
        start=time.perf_counter()
        try:
            req=urllib.request.Request(a.url.rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=1200) as resp:response=json.load(resp)
            row['response']=response
            choice=response['choices'][0]
            answer=choice['message'].get('content') or ''
            row.update(answer=answer,finish_reason=choice.get('finish_reason'),truncated=choice.get('finish_reason')=='length')
            row['ok']=answer.strip().rstrip('。.!！')==case['expected'] and not row['truncated']
        except Exception as error:row.update(ok=False,error=repr(error))
        row['seconds']=time.perf_counter()-start
        (out/f'{phase}-{case["name"]}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in ('response','messages')},ensure_ascii=False),flush=True)
        return row
    rows=[]
    for phase,concurrency,subset in [('sequential',1,cases[:4]),('concurrent',4,cases)]:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:rows.extend(pool.map(probe,[(phase,c) for c in subset]))
    summary={'total':len(rows),'passed':sum(r['ok'] for r in rows),'all_passed':all(r['ok'] for r in rows),
             'scope':'known-answer retrieval with1024/8192/16384-token prompts; not a general long-context benchmark'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return 0 if summary['all_passed'] else 2

if __name__=='__main__':raise SystemExit(main())
