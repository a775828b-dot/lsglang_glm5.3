"""lkqmoe startup hook for the GLM-5.3 Flash NVFP4 pipeline; everything is enabled by the launcher."""
import os
if os.environ.get('LKQMOE_MAIN_CPUS'):
    # Keep Python/scheduler threads off the pinned CPU expert workers; the workers and the
    # mailbox dispatcher set their own affinity.
    cpus=set()
    for part in os.environ['LKQMOE_MAIN_CPUS'].split(','):
        lo,_,hi=part.partition('-');cpus.update(range(int(lo),int(hi or lo)+1))
    os.sched_setaffinity(0,cpus&os.sched_getaffinity(0) or os.sched_getaffinity(0))
if os.environ.get('LKQMOE_MODE') in ('shadow','replace','standalone'):
    from lkqmoe.integration import install
    install()
if os.environ.get('LKQMOE_SMALL_M_GEMM') == '1':
    # BF16 GEMM for decode-sized batches (compiled module, used by fp8_weights for unconverted layers too)
    from lkqmoe.gpu.small_gemm import install as install_small_gemm
    install_small_gemm()
if os.environ.get('LKQMOE_GLM53_MHC') == '1':
    from lkqmoe.gpu.glm53_mhc import install as install_glm53_mhc
    install_glm53_mhc()
if os.environ.get('LKQMOE_GLM53_CPU_IMAGE') == '1':
    from lkqmoe.gpu.glm53_cpu_image import install as install_glm53_cpu_image
    install_glm53_cpu_image()
if os.environ.get('LKQMOE_INDEXER_Q_CHUNK'):
    from lkqmoe.gpu.indexer_chunk import install as install_indexer_chunk
    install_indexer_chunk()
if os.environ.get('LKQMOE_FP8_WEIGHTS') == '1':
    from lkqmoe.gpu.fp8_weights import install as install_fp8_weights
    install_fp8_weights()
if os.environ.get('LKQMOE_RESIDENT_W4A16') == '1':
    from lkqmoe.gpu.resident_w4a16 import install as install_resident_w4a16
    install_resident_w4a16()
