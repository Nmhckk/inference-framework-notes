import importlib.util,json,time
from pathlib import Path
import torch,torch_npu
root=Path('/work/kpool-optimization')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ref=load('original',root/'glm5_next_kpool.original.py').pooled_states
Reuse=load('reuse',root/'pool_reuse.py').VerifiedHistoryPoolReuse
torch.npu.set_device(0);torch.manual_seed(17);results=[]
for batch,base in [(1,129),(4,129),(8,81920)]:
 ape=torch.randn(4,128,device='npu',dtype=torch.bfloat16);history=torch.randn(batch,base+24,257,device='npu',dtype=torch.bfloat16);history[:,:,-1]=1;reuse=Reuse(ref)
 for step,(length,mode) in enumerate([(base,'new'),(base+1,'append'),(base+3,'mtp'),(base+2,'rollback'),(base+6,'append'),(base+7,'overwrite'),(base+8,'reorder'),(base+9,'ape'),(base+10,'leftpad'),(base+11,'invalid')]):
  if mode=='overwrite':history[:,10,0]+=1
  if mode=='reorder':history=history.flip(0).clone()
  if mode=='ape':ape.add_(.01)
  if mode=='leftpad':history[:,:5,-1]=0
  if mode=='invalid':history[:,:,-1]=0
  x=history[:,:length];valid=x[:,:,-1]>0;expected=ref(x,valid,ape,128,4);actual=reuse(x,valid,ape,128,4)
  for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
 history[:,:,-1]=1;x=history[:,:base];valid=x[:,:,-1]>0;reuse=Reuse(ref);reuse(x,valid,ape,128,4)
 row={'batch':batch,'length':base,'cases_passed':10,'bit_exact':True}
 for name,fn in [('reference',ref),('reuse',reuse)]:
  torch.npu.synchronize();start=time.perf_counter()
  for _ in range(5):fn(x,valid,ape,128,4)
  torch.npu.synchronize();row[name+'_ms']=(time.perf_counter()-start)*200
 row.update(hits=reuse.hits,misses=reuse.misses);print(json.dumps(row),flush=True);results.append(row)
(root/'pool-reuse-validation.json').write_text(json.dumps(results,indent=2))
