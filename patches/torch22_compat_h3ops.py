"""Bridge ALL comfy_kitchen eager implementations onto torch.ops.comfy_kitchen.*

The on-device comfy_kitchen CUDA extension was built with a reduced op set
(int8_linear, rms_rope only). ComfyUI's quantized-model paths invoke the rest
via torch.ops.comfy_kitchen.<op>(...), which would raise AttributeError.

comfy_kitchen.backends.eager ships pure-python implementations for every op.
Two subtleties handled here:

1. Plain name bridging: any missing op in eager.__all__ is registered directly.
2. dtype-code adapters: the top-level comfy_kitchen wrappers convert torch.dtype
   to an integer code before calling torch.ops (DTYPE_TO_CODE), but several eager
   implementations expect a torch.dtype. For those ops we interpose an adapter
   that maps int codes back to dtypes (DTYPE_CODE_TO_DTYPE: 0=f32 1=f16 2=bf16
   5=e4m3fn 6=e5m2).
3. quantize_fp8 / dequantize_fp8 have no same-name eager function; they map to
   quantize_per_tensor_fp8 / dequantize_per_tensor_fp8.

Ops already provided by the compiled extension are left untouched.
Run at interpreter startup via torch22_compat_h3ops.pth.
"""

_reported = False
_bridged_names = []


def _code_to_dtype(x):
    import torch
    if isinstance(x, int):
        from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
        return DTYPE_CODE_TO_DTYPE.get(x, torch.float32)
    return x


def _bridge_all():
    global _reported, _bridged_names
    try:
        import torch
        import comfy_kitchen  # noqa: F401
        from comfy_kitchen.backends import eager
        from comfy_kitchen.backends.eager import quantization as eq
    except Exception:
        return

    ns = torch.ops.comfy_kitchen

    def register(name, fn):
        if hasattr(ns, name):
            return False
        setattr(ns, name, fn)
        _bridged_names.append(name)
        return True

    # ---- dtype-code adapters (top-level wrappers pass int codes) ----
    if hasattr(eager, "dequantize_nvfp4"):
        def _dequantize_nvfp4(qx, per_tensor_scale, block_scales, output_type, hi_first=True,
                              _orig=eager.dequantize_nvfp4):
            return _orig(qx, per_tensor_scale, block_scales, _code_to_dtype(output_type), hi_first)

        register("dequantize_nvfp4", _dequantize_nvfp4)

    if hasattr(eager, "scaled_mm_nvfp4"):
        def _scaled_mm_nvfp4(a, b, tsa, tsb, bsa, bsb, bias=None, out_dtype=None, alpha=None,
                             _orig=eager.scaled_mm_nvfp4):
            return _orig(a, b, tsa, tsb, bsa, bsb, bias, _code_to_dtype(out_dtype), alpha)

        register("scaled_mm_nvfp4", _scaled_mm_nvfp4)

    if hasattr(eager, "dequantize_mxfp8"):
        def _dequantize_mxfp8(qx, block_scales, output_type, _orig=eager.dequantize_mxfp8):
            return _orig(qx, block_scales, _code_to_dtype(output_type))

        register("dequantize_mxfp8", _dequantize_mxfp8)

    if hasattr(eager, "scaled_mm_mxfp8"):
        def _scaled_mm_mxfp8(a, b, bsa, bsb, bias=None, out_dtype=None,
                             _orig=eager.scaled_mm_mxfp8):
            return _orig(a, b, bsa, bsb, bias, _code_to_dtype(out_dtype))

        register("scaled_mm_mxfp8", _scaled_mm_mxfp8)

    # ---- name-mapped fp8 helpers (no same-name eager function) ----
    if hasattr(eq, "quantize_per_tensor_fp8"):
        def _quantize_fp8(x, scale, output_type, _orig=eq.quantize_per_tensor_fp8):
            return _orig(x, scale, _code_to_dtype(output_type))

        register("quantize_fp8", _quantize_fp8)

    if hasattr(eq, "dequantize_per_tensor_fp8"):
        def _dequantize_fp8(x, scale, output_type, _orig=eq.dequantize_per_tensor_fp8):
            return _orig(x, scale, _code_to_dtype(output_type))

        register("dequantize_fp8", _dequantize_fp8)

    # ---- generic: every other eager op missing from the compiled set ----
    for name in getattr(eager, "__all__", []):
        fn = getattr(eager, name, None)
        if fn is not None:
            register(name, fn)

    if _bridged_names and not _reported:
        print(
            "[compat] torch 2.2 comfy_kitchen eager bridge (h3ops): %d ops bridged -> %s"
            % (len(_bridged_names), sorted(_bridged_names)[:8]),
            flush=True,
        )
        _reported = True


_bridge_all()


def _patch_nvfp4_linear_dtype():
    """nvfp4 linear slow-path dequantizes weights to bf16 but leaves fp32
    activations untouched -> F.linear dtype mismatch on torch 2.2. Homogenize
    dtypes around the dequantized weight."""
    try:
        import torch
        from comfy_kitchen.tensor import nvfp4 as nv
        from comfy_kitchen.tensor import base as tb
        from comfy_kitchen.tensor.base import QuantizedTensor
    except Exception:
        return
    orig = nv._handle_nvfp4_linear

    def fixed(qt, args, kwargs):
        input_tensor, weight = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
        if not (isinstance(input_tensor, QuantizedTensor) and isinstance(weight, QuantizedTensor)):
            dq = list(nv.dequantize_args((input_tensor, weight, bias)))
            anchor = dq[1].dtype
            if dq[0].dtype != anchor:
                dq[0] = dq[0].to(anchor)
            if len(dq) > 2 and dq[2] is not None and dq[2].dtype != anchor:
                dq[2] = dq[2].to(anchor)
            return torch.nn.functional.linear(*dq)
        return orig(qt, args, kwargs)

    nv._handle_nvfp4_linear = fixed
    replaced = 0
    for op, handlers in list(getattr(tb, "_LAYOUT_DISPATCH_TABLE", {}).items()):
        for cls, fn in list(handlers.items()):
            if fn is orig:
                handlers[cls] = fixed
                replaced += 1
    print(f"[compat] nvfp4 linear dtype patch applied ({replaced} dispatch slots)", flush=True)


_patch_nvfp4_linear_dtype()
