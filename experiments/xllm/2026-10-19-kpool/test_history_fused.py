"""Numerical and microperformance validation on an isolated healthy NPU."""
import importlib.util,json,time
from pathlib import Path
import torch,torch_npu

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
base=Path('/work/kpool-optimization')
ref=load('reference_pool',base/'glm5_next_kpool.original.py').pooled_states
new=load('fused_pool',base/'kpool_history_fused.py').pooled_states_fused
torch.npu.set_device(0);torch.manual_seed(47)
results=[]
for batch,length,kind in [(1,17,'tail'),(4,129,'holes'),(3,128,'leftpad'),(2,32,'empty'),(1,81920,'full'),(8,82534,'full')]:
 x=torch.randn(batch,length,257,device='npu',dtype=torch.bfloat16);ape=torch.randn(4,128,device='npu',dtype=torch.bfloat16);valid=torch.ones(batch,length,device='npu',dtype=torch.bool)
 if kind=='holes':valid[:,::5]=False
 if kind=='leftpad':valid[:,:7]=False;valid[:,-5:]=False
 if kind=='empty':valid[:]=False
 expect=ref(x,valid,ape,128,4);actual=new(x,valid,ape,128,4);torch.npu.synchronize()
 torch.testing.assert_close(actual[1],expect[1],rtol=0,atol=0);torch.testing.assert_close(actual[2],expect[2],rtol=0,atol=0)
 diff=(actual[0].float()-expect[0].float()).abs();maxdiff=diff.max().item();neq=(actual[0]!=expect[0]).sum().item()
 torch.testing.assert_close(actual[0],expect[0],rtol=0.008,atol=0.008)
 timing={}
 for label,fn in [('reference',ref),('fused',new)]:
  fn(x,valid,ape,128,4);torch.npu.synchronize();torch.npu.reset_peak_memory_stats();start_mem=torch.npu.memory_allocated();start=time.perf_counter()
  for _ in range(5):out=fn(x,valid,ape,128,4)
  torch.npu.synchronize();timing[label]={'ms':(time.perf_counter()-start)*200,'peak_extra_mib':(torch.npu.max_memory_allocated()-start_mem)/2**20}
 record={'batch':batch,'length':length,'kind':kind,'max_abs_error':maxdiff,'different_elements':neq,'elements':actual[0].numel(),'timing':timing};results.append(record);print(json.dumps(record),flush=True)
(base/'microbenchmark.json').write_text(json.dumps(results,indent=2))
