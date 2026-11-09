"""Load the matching installed LightOp GLM entrypoints without JIT or overrides."""

from functools import lru_cache


@lru_cache(maxsize=None)
def lightop_glm_op(name):
    try:
        from lightop import op
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GLM optimizations require the matching LightOp build"
        ) from exc
    if getattr(op, "GLM53_OPT_ABI_VERSION", None) != 1:
        raise RuntimeError("LightOp GLM53_OPT_ABI_VERSION=1 is required")
    function = getattr(op, name, None)
    if not callable(function):
        raise RuntimeError(f"Installed LightOp lacks the required GLM operator: {name}")
    return function
