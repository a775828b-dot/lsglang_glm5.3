"""Opt-in (LKQMOE_GLM53_CPU_IMAGE=1): decode GLM-5.3 images on the CPU.

sglang's multimodal processors decode strict JPEGs with nvJPEG on the GPU by default
(`gpu_image_decode = True`). The GLM-5.x processor copies the decoded tensor straight back to
the CPU for its native preprocessing, so the GPU decode only costs device memory (the long-context
VRAM margin) and a round trip. This sets `gpu_image_decode = False` on the GLM image processor
classes only; with `--image-processor-backend pil` the whole image path stays on the CPU, as in
the Qwen3.8 pipeline. Pixels are identical up to the JPEG decoder (PIL instead of nvJPEG).
"""
import importlib.abc
import importlib.machinery
import sys

_TARGET = 'sglang.srt.multimodal.processors.glm4v'
_CLASSES = ('Glm4vImageProcessor', 'Glm5NextImageProcessor')


def patch(module):
    if getattr(module, '_lkqmoe_glm53_cpu_image', False):
        return
    found = [name for name in _CLASSES if hasattr(module, name)]
    if not found:
        raise RuntimeError('Unsupported sglang GLM multimodal processor interface')
    for name in found:
        getattr(module, name).gpu_image_decode = False
    module._lkqmoe_glm53_cpu_image = True


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate):
        self.delegate = delegate

    def create_module(self, spec):
        fn = getattr(self.delegate, 'create_module', None)
        return fn(spec) if fn else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


def install():
    if _TARGET in sys.modules:
        patch(sys.modules[_TARGET])
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
