import torch,ctypes,json,hashlib,pathlib
root=pathlib.Path('/qualification/preflight/publication-20260917')
lib=ctypes.CDLL(str(root/'head-current.so'));f=lib.run
f.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]; f.restype=ctypes.c_int
prop=torch.cuda.get_device_properties(0);assert prop.gcnArchName.startswith('gfx1201')
cu=prop.multi_processor_count; torch.manual_seed(42)
w=torch.randn(248320,5120,device='cuda',dtype=torch.bfloat16)
def run(x):
 y=torch.empty((len(x),248320),device='cuda',dtype=torch.bfloat16)
 assert f(w.data_ptr(),x.data_ptr(),y.data_ptr(),len(x),cu,torch.cuda.current_stream().cuda_stream)==0
 return y
cases=[]
for scale in (0.0,0.01,1.0,16.0):
 x=torch.randn(8,5120,device='cuda',dtype=torch.bfloat16)*scale
 m1=torch.cat([run(row[None]) for row in x]);m4=torch.cat([run(part) for part in x.split(4)]);m8=run(x)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph): captured=run(x)
 graph.replay()
 item={'scale':scale,'m1_m8_differences':int((m1!=m8).sum()),'m4_m8_differences':int((m4!=m8).sum()),'graph_differences':int((captured!=m8).sum())}
 cases.append(item)
 assert not any(item[k] for k in item if k.endswith('differences'))
result={'scope':'Current upstream kernel extraction, BF16 arithmetic helpers and native launch; full vLLM dispatch/build integration untested','cases':cases,'positions':32,'compared_values_per_comparison':32*248320,'source_sha256':hashlib.sha256((root/'head-current.hip').read_bytes()).hexdigest(),'gpu':prop.name,'compute_units':cu}
(root/'head-result-001.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
