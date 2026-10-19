"""Offline GSM8K regression over an OpenAI-compatible HTTP endpoint.

Custom transparent harness, not an EvalScope run. Reports exact numeric answers
from final content only; reasoning-only/truncated output cannot pass.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random
import re
import threading
import time
import urllib.request


def numeric(text):
    try:
        return str(Decimal(text.replace(',', '').strip()).normalize())
    except InvalidOperation:
        return None


def extract(text):
    marked = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if marked:
        return numeric(marked[-1])
    boxed = re.findall(r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}', text)
    return numeric(boxed[-1]) if boxed else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://172.16.12.11:28888/v1')
    p.add_argument('--data', default='/work/gsm8k-test.jsonl')
    p.add_argument('--train', default='/work/gsm8k-train.jsonl')
    p.add_argument('--limit', type=int, default=64)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--output', required=True)
    p.add_argument('--selection-seed', type=int, default=20261019)
    p.add_argument('--few-shot', type=int, default=4)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    a = p.parse_args()
    data_path, train_path = Path(a.data), Path(a.train)
    data = [json.loads(x) for x in data_path.read_text().splitlines()]
    train = [json.loads(x) for x in train_path.read_text().splitlines()]
    indices = random.Random(a.selection_seed).sample(range(len(data)), min(a.limit, len(data)))
    run_dir = Path(a.output)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(vars(a), dataset='GSM8K main/test', split_count=len(data), indices=indices,
                    data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                    source='https://github.com/openai/grade-school-math',
                    scorer='final-content-only #### or boxed numeric exact match',
                    harness='stdlib custom, not EvalScope', started_at=time.time())
    (run_dir/'manifest.json').write_text(json.dumps(manifest, indent=2))
    examples = '\n\n'.join('Question: '+x['question']+'\nAnswer: '+x['answer'] for x in train[:a.few_shot])
    lock = threading.Lock()
    def probe(index):
        item = data[index]
        prompt = ('Solve the math word problem. Show your reasoning concisely, then end your final answer with '
                  '#### followed by the numeric answer.\n\n'+examples+'\n\nQuestion: '+item['question']+'\nAnswer:')
        payload = {'model':'glm53-flash', 'messages':[{'role':'user','content':prompt}],
                   'max_tokens':a.max_tokens, 'temperature':a.temperature, 'top_p':a.top_p,
                   'seed':42, 'stream':False, 'chat_template_kwargs':{'enable_thinking':True}}
        result = {'index':index, 'question':item['question'], 'reference':item['answer'],
                  'expected':extract(item['answer']), 'request':payload}
        start = time.perf_counter()
        try:
            req = urllib.request.Request(a.url.rstrip('/')+'/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=1200) as resp:
                body = json.load(resp)
            result['response'] = body
            choice = body['choices'][0]
            content = choice['message'].get('content') or ''
            result.update(predicted=extract(content), finish_reason=choice.get('finish_reason'),
                          has_answer=bool(content.strip()))
            result['correct'] = result['predicted'] == result['expected'] and bool(content.strip())
            result['truncated'] = choice.get('finish_reason') == 'length'
        except Exception as error:
            result.update(correct=False, error=repr(error), truncated=False)
        result['seconds'] = time.perf_counter()-start
        (run_dir/f'item-{index:04d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        with lock:
            with (run_dir/'progress.jsonl').open('a') as out:
                out.write(json.dumps({k:v for k,v in result.items() if k not in ('request','response','reference','question')},ensure_ascii=False)+'\n')
            print(json.dumps({k:result.get(k) for k in ('index','correct','predicted','expected','truncated','error','seconds')}, ensure_ascii=False), flush=True)
        return result
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results = list(pool.map(probe, indices))
    correct=sum(r['correct'] for r in results)
    errors=sum('error' in r for r in results)
    truncations=sum(r['truncated'] for r in results)
    # This predeclared regression gate is not a claim of complete model accuracy.
    gate=correct/len(results)>=0.95 and errors==0 and truncations==0
    report={'n':len(results), 'correct':correct, 'accuracy':correct/len(results),
            'errors':errors, 'truncations':truncations, 'seconds':time.perf_counter()-started,
            'regression_gate':gate, 'gate_rule':'accuracy>=95%, zero transport errors, zero truncations',
            'scope':'sampled GSM8K regression; not full test split or cross-framework reference',
            'items':[{k:r.get(k) for k in ('index','correct','predicted','expected','truncated','error')} for r in results]}
    (run_dir/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if gate else 2


if __name__ == '__main__':
    raise SystemExit(main())
