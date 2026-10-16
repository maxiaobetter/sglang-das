import asyncio
from http import HTTPStatus
from types import MethodType, SimpleNamespace

import fastapi
import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(disaggregation_mode: DisaggregationMode):
    manager = SimpleNamespace(
        disaggregation_mode=disaggregation_mode,
        rid_to_state={"rid": object()},
        enable_lora=False,
    )
    manager._drop_aborted_state = MethodType(
        TokenizerManager._drop_aborted_state, manager
    )
    return manager


def _state():
    return SimpleNamespace(obj=SimpleNamespace(rid="rid", lora_path=None))


def _abort_out(status_code):
    finish_reason = {"type": "abort", "message": "mm assemble failed"}
    if status_code is not None:
        finish_reason["status_code"] = status_code
    return {"meta_info": {"finish_reason": finish_reason}}


def _handle(manager, out, *, is_stream):
    return asyncio.run(
        TokenizerManager._handle_abort_finish_reason(
            manager, out, _state(), is_stream=is_stream
        )
    )


@pytest.mark.parametrize(
    "status_code, expected",
    [
        (None, HTTPStatus.INTERNAL_SERVER_ERROR),
        (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.REQUEST_TIMEOUT),
    ],
)
def test_pd_prefill_abort_is_never_http_200(status_code, expected):
    """A PD prefill abort without one of the legacy error codes used to return
    HTTP 200, so the router never aborted the paired decode, which then waited
    for KV until the disaggregation timeout."""
    manager = _manager(DisaggregationMode.PREFILL)

    with pytest.raises(fastapi.HTTPException) as exc_info:
        _handle(manager, _abort_out(status_code), is_stream=False)

    assert exc_info.value.status_code == expected
    assert exc_info.value.detail == "mm assemble failed"
    assert "rid" not in manager.rid_to_state


def test_graceful_abort_outside_pd_prefill_keeps_normal_response():
    manager = _manager(DisaggregationMode.NULL)

    assert _handle(manager, _abort_out(None), is_stream=False) is None
    assert "rid" in manager.rid_to_state


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
