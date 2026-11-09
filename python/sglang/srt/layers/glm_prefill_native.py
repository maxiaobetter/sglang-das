"""Load installed LightOp operators with the required optional kernel control."""

from functools import lru_cache


@lru_cache(maxsize=None)
def lightop_glm_op(name, option):
    try:
        from lightop import op
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "GLM optimizations require the matching LightOp build"
        ) from exc
    function = getattr(op, name, None)
    if not callable(function) or f"{option}: bool = False" not in (
        function.__doc__ or ""
    ):
        raise RuntimeError(
            f"Installed LightOp must support {name}(..., {option}=False)"
        )
    return function
