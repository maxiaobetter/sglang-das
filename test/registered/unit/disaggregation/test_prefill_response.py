import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from sglang.srt.disaggregation.encoder import prefill_response
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    StreamOptions,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(
    params=[
        pytest.param(("prefill", True, True), id="epd-prefill"),
        pytest.param(("prefill", False, False), id="pd-prefill"),
        pytest.param(("decode", True, False), id="epd-decode"),
        pytest.param(("decode", False, False), id="pd-decode"),
        pytest.param(("null", True, False), id="language-only-without-pd"),
        pytest.param(("null", False, False), id="without-disaggregation"),
    ]
)
def deployment(request, monkeypatch):
    mode, language_only, forces_non_streaming = request.param
    monkeypatch.setattr(
        prefill_response,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_mode=mode, language_only=language_only),
    )
    return forces_non_streaming


def _request(kind, stream=True):
    kwargs = {
        "stream": stream,
        "stream_options": StreamOptions(include_usage=True) if stream else None,
    }
    if kind == "chat":
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}], **kwargs
        )
    return CompletionRequest(prompt="hello", **kwargs)


@pytest.mark.parametrize("kind", ["chat", "completion"])
@pytest.mark.parametrize("stream", [True, False])
def test_openai_response_policy(deployment, kind, stream):
    request = _request(kind, stream)
    before = request.model_dump()

    prefill_response.drop_streaming_for_epd_prefill(request)

    expected = dict(before)
    if deployment:
        expected.update(stream=False, stream_options=None)
    assert request.model_dump() == expected
    assert prefill_response.epd_prefill_forces_non_streaming() is deployment


def test_embedding_request_is_unchanged(deployment):
    request = EmbeddingRequest(input="hello", dimensions=128)
    before = request.model_dump()

    prefill_response.drop_streaming_for_epd_prefill(request)

    assert request.model_dump() == before
    assert not hasattr(request, "stream")
    assert not hasattr(request, "stream_options")


def test_unrelated_streaming_request_is_unchanged(deployment):
    request = SimpleNamespace(stream=True, stream_options={"include_usage": True})
    before = vars(request).copy()

    prefill_response.drop_streaming_for_epd_prefill(request)

    assert vars(request) == before


async def _offload(func):
    return func()


class _Serving(OpenAIServingBase):
    """Exercise the real base handler and conversion order without inference."""

    def __init__(self, *, fail=False):
        super().__init__(
            SimpleNamespace(
                server_args=SimpleNamespace(),
                request_logger=SimpleNamespace(log_requests=False),
                run_tokenizer_offload=_offload,
            )
        )
        self.fail = fail
        self.converted_stream = None
        self.converted_stream_options = None

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        self.converted_stream = request.stream
        self.converted_stream_options = request.stream_options
        return GenerateReqInput(text="hello", stream=request.stream), request

    async def _handle_streaming_request(self, adapted_request, request, raw_request):
        async def events():
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        if self.fail:
            raise HTTPException(status_code=500, detail="prefill assembly failed")
        return JSONResponse({"text": "ok"})


@pytest.mark.parametrize("kind", ["chat", "completion"])
def test_openai_entrypoint_applies_policy_before_conversion(deployment, kind):
    request = _request(kind)
    original_options = request.stream_options
    serving = _Serving()

    response = asyncio.run(serving.handle_request(request, None))

    assert serving.converted_stream is (not deployment)
    if deployment:
        assert serving.converted_stream_options is None
        assert isinstance(response, JSONResponse)
    else:
        assert serving.converted_stream_options is original_options
        assert isinstance(response, StreamingResponse)


@pytest.mark.parametrize("kind", ["chat", "completion"])
def test_openai_entrypoint_returns_http_500_for_epd_prefill_failure(deployment, kind):
    response = asyncio.run(_Serving(fail=True).handle_request(_request(kind), None))

    if deployment:
        assert response.status_code == 500
        assert b"prefill assembly failed" in response.body
        assert "text/event-stream" not in response.headers["content-type"]
    else:
        assert isinstance(response, StreamingResponse)


def _native_manager(*, fail=False):
    async def generate_request(obj, request):
        if fail:
            raise HTTPException(status_code=500, detail="prefill assembly failed")
        yield {"text": "ok"}

    return SimpleNamespace(
        generate_request=generate_request,
        create_abort_task=lambda obj: None,
    )


@pytest.fixture
def native_server(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES", "false")
    state = SimpleNamespace(tokenizer_manager=_native_manager())
    monkeypatch.setattr(http_server, "_global_state", state)
    return state


def test_native_generate_entrypoint_applies_policy(deployment, native_server):
    request = GenerateReqInput(text="hello", stream=True)

    response = asyncio.run(http_server.generate_request(request, None))

    assert request.stream is (not deployment)
    if deployment:
        assert response.status_code == 200
        assert b'"text":"ok"' in response.body
    else:
        assert isinstance(response, StreamingResponse)


def test_native_generate_entrypoint_propagates_epd_prefill_failure(
    deployment, native_server
):
    native_server.tokenizer_manager = _native_manager(fail=True)
    request = GenerateReqInput(text="hello", stream=True)

    if deployment:
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(http_server.generate_request(request, None))
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "prefill assembly failed"
    else:
        response = asyncio.run(http_server.generate_request(request, None))
        assert isinstance(response, StreamingResponse)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
