"""Minimal torch-free reader for PyG Data objects saved with torch.save (zip format)."""
import pickle, zipfile, numpy as np, types, sys

_DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64,
       'IntStorage': np.int32, 'BoolStorage': np.bool_, 'ByteStorage': np.uint8, 'HalfStorage': np.float16}

class _Stub:
    def __init__(self, *a, **k): self.__dict__['_a'] = a
    def __setstate__(self, s):
        if isinstance(s, dict): self.__dict__.update(s)
        else: self.__dict__['_state'] = s

def _rebuild_tensor(storage, offset, size, stride, *a):
    arr = storage
    n = int(np.prod(size)) if len(size) else 1
    if len(size) == 0: return arr[offset]
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=size,
        strides=[s * arr.itemsize for s in stride]).copy()

def load(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == '_rebuild_tensor_v2': return _rebuild_tensor
            if name in _DT: return name
            if mod.startswith('collections') and name == 'OrderedDict':
                import collections; return collections.OrderedDict
            if name == '_rebuild_parameter': return lambda d, *a: d
            try: return super().find_class(mod, name)
            except Exception: return type(name, (_Stub,), {})
        def persistent_load(self, pid):
            _, stype, key, loc, numel = pid
            raw = z.read(f'{root}/data/{key}')
            return np.frombuffer(raw, dtype=_DT[stype])[:numel]
    return U(z.open(f'{root}/data.pkl')).load()

def to_dict(obj):
    d = obj.__dict__
    for k in ('_store', '_mapping'):
        if k in d: d = d[k] if not hasattr(d[k], '__dict__') else to_dict(d[k])
    if isinstance(d, dict) and '_mapping' in d: d = d['_mapping']
    return d
