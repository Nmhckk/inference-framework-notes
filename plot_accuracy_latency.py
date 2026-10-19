"""Plot observed client latency during accuracy traffic, without claiming perf."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter

root=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (root/'gsm8k-predictions.jsonl').read_text().splitlines()]
values=sorted(r['seconds'] for r in rows)
fig,ax=plt.subplots(figsize=(7.3,4.3))
ax.step(values,[(i+1)/len(values) for i in range(len(values))],where='post',color='#1769a8')
ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator([20,50,100,200,500,1000]));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel('Client end-to-end latency (seconds, log scale)');ax.set_ylabel('Fraction of completed requests')
ax.set_title('GSM8K64 mixed-traffic latency observation');ax.grid(alpha=.2)
ax.annotate(f'Longest: {max(values):.1f}s\n137 output tokens, correct answer',(max(values),1),xytext=(-170,-55),textcoords='offset points',arrowprops={'arrowstyle':'->'})
fig.text(.5,.01,'Accuracy traffic overlapped retrieval checks; not a fixed-length performance benchmark.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(root/'accuracy-latency-observation.png',dpi=180);plt.close(fig)
