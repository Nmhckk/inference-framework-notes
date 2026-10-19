import json,time,torch,torch_npu
from pathlib import Path

def reference(cache,bt,lens):
 bs=cache.shape[1];width=cache.shape[-1];b=bt.shape[0];klens=lens.to(torch.int64);mx=int(klens.max().item());flat=cache.view(-1,width);bt=bt.to(torch.int64);out=torch.zeros(b,mx,width,dtype=cache.dtype,device=cache.device)
 for i in range(b):
  kl=int(klens[i].item())
  if not kl:continue
  full,tail=divmod(kl,bs);rows=[]
  if full:rows.append(flat.index_select(0,(bt[i,:full,None]*bs+torch.arange(bs,device=cache.device)).reshape(-1)))
  if tail:rows.append(flat.index_select(0,int(bt[i,full].item())*bs+torch.arange(tail,device=cache.device)))
  out[i,:kl]=torch.cat(rows) if len(rows)>1 else rows[0]
 return out

def vectorized(cache,bt,lens):
 bs=cache.shape[1];width=cache.shape[-1];b=bt.shape[0];lens=lens.to(torch.int64);mx=int(lens.max().item());pos=torch.arange(mx,device=cache.device);live=pos[None,:]<lens[:,None];physical=bt.to(torch.int64).index_select(1,pos//bs);slots=torch.where(live,physical*bs+pos[None,:]%bs,0);return cache.view(-1,width).index_select(0,slots.reshape(-1)).view(b,mx,width).masked_fill(~live[:,:,None],0)

torch.npu.set_device(0);torch.manual_seed(7);results=[]
for b,length in [(1,129),(8,129),(8,82534)]:
 bs=128;blocks=(length+bs-1)//bs;cache=torch.randn(b*blocks,bs,1,257,device='npu',dtype=torch.bfloat16);bt=torch.randperm(b*blocks,device='npu').reshape(b,blocks);lens=torch.tensor([max(0,length-i*17) for i in range(b)],device='npu')
 if length==129 and b>1:lens[-1]=0;bt[-1]=-1
 a=reference(cache,bt,lens);z=vectorized(cache,bt,lens);torch.testing.assert_close(a,z,rtol=0,atol=0);row={'batch':b,'length':length,'bit_exact':True}
 for name,fn in [('reference',reference),('vectorized',vectorized)]:
  fn(cache,bt,lens);torch.npu.synchronize();start=time.perf_counter()
  for _ in range(8):fn(cache,bt,lens)
  torch.npu.synchronize();row[name+'_ms']=(time.perf_counter()-start)*125
 print(json.dumps(row),flush=True);results.append(row)
Path('/work/kpool-optimization/gather-microbenchmark.json').write_text(json.dumps(results,indent=2))
