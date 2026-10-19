"""CPU-only metadata routing probe of the actual runner method, not NPU kernels."""
import ast
import argparse
import difflib
import json
from pathlib import Path
from types import SimpleNamespace as NS

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent.parent / 'local-restore-check/xllm/xllm/python/model_executor/runners/decode_acl_graph.py')
SOURCE=parser.parse_args().source

class Vector:
    def __init__(self, data, start=0, stop=None):
        self.data, self.start, self.stop = data, start, len(data) if stop is None else stop
        self.device = 'cpu'
    def numel(self): return self.stop-self.start
    def __getitem__(self, key):
        assert isinstance(key,slice)
        return Vector(self.data,self.start+(key.start or 0),self.start+(key.stop if key.stop is not None else self.numel()))
    def copy_(self, other): self.data[self.start:self.stop]=other.data[other.start:other.stop]
    def zero_(self): self.data[self.start:self.stop]=[0]*self.numel()
    def to(self, *args): return self
    def contiguous(self): return self


def invoke(source, is_mla, has_linear):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DecodeAclGraphRunner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_fill_entry')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),method],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace={'torch':NS(int32=object()),'kernels':NS(update_decode_graph_metadata=lambda *args:None),'resolve_expanded_decode_metadata':lambda metadata:None}
    exec(compile(module,str(SOURCE),'exec'),namespace)
    runner=NS(attention_backend=NS(is_mla=is_mla),_has_linear_cache=has_linear)
    runner._effective_slot_mapping=lambda *args:Vector([1,2,3,4])
    runner._decode_metadata=lambda *args:(NS(shape=(4,1)),None,None,None,None,None)
    runner._validate_decode_token_layout=lambda *args:None
    runner._cumulative_lengths=lambda *args:None
    runner._fill_host_metadata=lambda *args:None
    runner._fill_new_cache_slots_host_metadata=lambda *args:None
    runner._fill_dsa_block_tables=lambda *args:None
    static=NS(multi_block_tables=[],slot_mapping=None,kv_cu_seq_lens=None,paged_kv_indptr=None,paged_kv_indices=None,paged_kv_last_page_len=None,linear_state_indices=Vector([0]*4),block_table=None,mega_moe_token_mask=None)
    entry=NS(batch_size=4,static_metadata=static,static_input_ids=None,static_positions=None,kv_seq_lens_delta=None,static_input_embedding=None)
    metadata=NS(kv_cu_seq_lens=None,linear_state_indices=Vector([3,7,11,19]))
    namespace['_fill_entry'](runner,entry,Vector([1]*4),Vector([100]*4),metadata,4,None)
    return static.linear_state_indices.data

source=SOURCE.read_text()
old='if getattr(self.attention_backend, "is_mla", False)\n            else getattr(metadata, "linear_state_indices", None)'
new='if getattr(self.attention_backend, "is_mla", False) and not self._has_linear_cache\n            else getattr(metadata, "linear_state_indices", None)'
assert source.count(old)==1
candidate=source.replace(old,new)
results={'scope':'Actual _fill_entry method with unrelated NPU metadata kernel stubbed; only CPU routing of state IDs is tested, not graph execution','expected_hybrid_state_ids':[3,7,11,19],'original':{},'proposed':{}}
for label,mla,linear in [('hybrid_KDA_DSA',True,True),('pure_MLA',True,False),('linear_non_MLA',False,True)]:
    results['original'][label]=invoke(source,mla,linear)
    results['proposed'][label]=invoke(candidate,mla,linear)
assert results['original']['hybrid_KDA_DSA']==[0]*4
assert results['proposed']['hybrid_KDA_DSA']==[3,7,11,19]
assert results['proposed']['pure_MLA']==results['original']['pure_MLA']==[0]*4
assert results['proposed']['linear_non_MLA']==results['original']['linear_non_MLA']==[3,7,11,19]
root=Path(__file__).resolve().parent
(root/'graph-metadata-probe.json').write_text(json.dumps(results,indent=2))
(root/'graph-linear-state-proposed.patch').write_text(''.join(difflib.unified_diff(source.splitlines(keepends=True),candidate.splitlines(keepends=True),fromfile='a/xllm/python/model_executor/runners/decode_acl_graph.py',tofile='b/xllm/python/model_executor/runners/decode_acl_graph.py')))
print(json.dumps(results,indent=2))
