"""Aggregate three independent benchmark runs; keep raw run TPS and pooled tails."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics


def quantile(values,q):
    data=sorted(values);pos=(len(data)-1)*q;lo=int(pos);hi=min(lo+1,len(data)-1)
    return data[lo]+(data[hi]-data[lo])*(pos-lo)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cells=[json.loads(p.read_text()) for p in sorted(source.glob('cell-*.json'))]
    assert len(cells)==18,'Need all2workloads x3concurrencies x3repeats'
    groups=defaultdict(list)
    for cell in cells:
        s=cell['summary'];assert s['successes']==s['requests']==8
        assert all(r['ok'] and r['usage']['completion_tokens']==128 for r in cell['requests'])
        groups[(s['input_target'],s['concurrency'])].append(cell)
    rows=[]
    for (input_target,concurrency),runs in sorted(groups.items()):
        assert {r['summary']['repeat'] for r in runs}=={0,1,2}
        requests=[r for run in runs for r in run['requests']]
        tps=[r['summary']['output_tps'] for r in runs]
        usage=[r['usage']['prompt_tokens'] for r in requests]
        row={'input_target':input_target,'output_tokens':128,'concurrency':concurrency,'n':len(requests),'runs':3,
             'input_actual_min':min(usage),'input_actual_max':max(usage),
             'output_tps_mean':statistics.mean(tps),'output_tps_stdev':statistics.stdev(tps),
             'output_tps_min':min(tps),'output_tps_max':max(tps),
             'output_tps_combined':sum(r['usage']['completion_tokens'] for r in requests)/sum(r['summary']['duration'] for r in runs),
             'total_tps_mean':statistics.mean(r['summary']['total_tps'] for r in runs),
             'ttft_p50':quantile([r['ttft'] for r in requests],.5),'ttft_p95':quantile([r['ttft'] for r in requests],.95),
             'latency_p50':quantile([r['latency'] for r in requests],.5),'latency_p95':quantile([r['latency'] for r in requests],.95),
             'tpot_p50_ms':quantile([r['tpot'] for r in requests],.5)*1000,'tpot_p95_ms':quantile([r['tpot'] for r in requests],.95)*1000,
             'user_gen_tps_median':statistics.median(1/r['tpot'] for r in requests),
             'visible_answer_requests':sum(r['ttfa'] is not None for r in requests),
             'ttfa_p50_nonnull':quantile([r['ttfa'] for r in requests if r['ttfa'] is not None],.5) if any(r['ttfa'] is not None for r in requests) else None,
             'ttfa_p95_nonnull':quantile([r['ttfa'] for r in requests if r['ttfa'] is not None],.95) if any(r['ttfa'] is not None for r in requests) else None,
             'max_chunk_gap':max(r['max_chunk_gap'] for r in requests),
             'output_tps_per_active_logical_device':statistics.mean(tps)/32,
             'output_tps_per_installed_logical_device':statistics.mean(tps)/64,
             'output_tps_per_allocated_host':statistics.mean(tps)/4}
        rows.append(row)
    (out/'performance-summary.json').write_text(json.dumps({'groups':rows,'scope':'fixed128 output including reasoning; conservative configuration; small-sample P95, no stable P99 claim','resources':{'hosts':4,'active_logical_devices':32,'installed_logical_devices':64}},indent=2))
    with (out/'performance-summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,sharey=True)
    for ax,target,color in zip(axes,(1024,8192),('#1769a8','#b94f20')):
        points=[r for r in rows if r['input_target']==target]
        x=[r['user_gen_tps_median'] for r in points];y=[r['output_tps_mean'] for r in points]
        ax.plot(x,y,color=color,alpha=.55,lw=1)
        ax.errorbar(x,y,yerr=[r['output_tps_stdev'] for r in points],fmt='o',capsize=4,color=color)
        for r,xx,yy in zip(points,x,y):ax.annotate(f'C={r["concurrency"]}',(xx,yy),xytext=(-8,7),ha='right',textcoords='offset points',fontsize=10)
        ax.set_title(f'~{target//1024}K input / 128 output');ax.set_xlabel('Median generated tokens/s per request');ax.grid(alpha=.2);ax.set_xlim(left=0);ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Aggregate output tokens/s (all P + D)')
    fig.suptitle('xLLM GLM-5.3-Flash W8A8: initial 2P2D baseline',fontsize=13)
    fig.text(.5,.01,'4 hosts / 32 active / 64 installed logical devices; batch1, graph off, MTP off. TPS error bars: SD over3 runs. No tuned-peak claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(out/'performance-curve.png',dpi=180);fig.savefig(out/'performance-curve.svg');plt.close(fig)
    # Show the worst measured nonempty-chunk gap, explicitly as an example rather than P99.
    worst_cell,worst=max(((c,r) for c in cells for r in c['requests']),key=lambda pair:pair[1]['max_chunk_gap'])
    fast=min((r for r in worst_cell['requests'] if r['request_id']!=worst['request_id']),key=lambda r:r['latency'])
    evidence={'selection':'largest measured max_chunk_gap; warmups excluded',
              'cell':worst_cell['summary'],'worst_request':worst,'comparison_request':fast,
              'scope':'one observed example, not a stable tail percentile; SSE chunk counts are not token counts'}
    (out/'stream-stall-example.json').write_text(json.dumps(evidence,indent=2))
    fig,ax=plt.subplots(figsize=(8,4.6))
    for request,label,color in [(worst,'Largest-gap sample','#b94f20'),(fast,'Fast sample, same cell','#1769a8')]:
        times=[0]+[e['t'] for e in request['events']]
        counts=list(range(len(times)))
        ax.step(times,counts,where='post',label=label,color=color,lw=1.6)
    event_index=max(range(1,len(worst['events'])),key=lambda i:worst['events'][i]['gap'] or 0)
    left=worst['events'][event_index-1]['t'];right=worst['events'][event_index]['t']
    ax.axvspan(left,right,color='#b94f20',alpha=.1)
    ax.text((left+right)/2,.55*len(worst['events']),f'{right-left:.1f}s gap',ha='center',color='#9e3a17')
    ax.set_xlabel('Seconds since request send');ax.set_ylabel('Nonempty SSE chunks received')
    c=worst_cell['summary'];ax.set_title(f'Streaming pause example: ~{c["input_target"]//1024}K / 128, C={c["concurrency"]}, run{c["repeat"]+1}')
    ax.grid(alpha=.2);ax.legend(loc='lower right');fig.text(.5,.01,'Both responses generated128 tokens. A chunk may contain multiple tokens; this is a selected example, not P99.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'stream-stall-example.png',dpi=180);plt.close(fig)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
