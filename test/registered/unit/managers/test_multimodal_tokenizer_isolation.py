"""Text offload must not change the multimodal processor's tokenizer state."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import tokenizer_manager as tm  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMultimodalTokenizerIsolation(unittest.TestCase):
    def setUp(self):
        backend = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3},
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = Whitespace()
        self.processor = SimpleNamespace(
            tokenizer=PreTrainedTokenizerFast(
                tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
            )
        )
        self.mm_processor = SimpleNamespace(_tokenizer=self.processor.tokenizer)
        self.manager = tm.TokenizerManager.__new__(tm.TokenizerManager)
        self.manager.server_args = SimpleNamespace()
        self.manager.model_config = SimpleNamespace(is_multimodal=True, hf_config=None)
        with (
            patch.dict(os.environ, {"SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": ""}),
            patch.multiple(
                tm,
                get_disagg=lambda: SimpleNamespace(language_model_only=False),
                get_serving=lambda: SimpleNamespace(
                    skip_tokenizer_init=False, enable_dynamic_batch_tokenizer=False
                ),
                import_processors=lambda *args: None,
                get_processor_wrapper=lambda: self.processor,
                determine_tensor_transport_mode=lambda: None,
                get_mm_processor=lambda *args, **kwargs: self.mm_processor,
            ),
        ):
            self.manager.init_tokenizer_and_processor()
        if self.manager.tokenizer_offload_executor is not None:
            self.addCleanup(self.manager.tokenizer_offload_executor.shutdown)

    def test_text_backend_is_independent_and_token_ids_match(self):
        text = self.manager.tokenizer
        mm = self.mm_processor._tokenizer
        self.assertIs(self.manager.processor, self.processor)
        self.assertIs(mm, self.processor.tokenizer)
        self.assertIsNot(text, mm)
        self.assertIsNot(text._tokenizer, mm._tokenizer)
        self.assertEqual(text.encode("hello world"), mm.encode("hello world"))

    def test_text_encoding_preserves_multimodal_padding_and_truncation(self):
        text = self.manager.tokenizer
        mm = self.mm_processor._tokenizer
        self.assertEqual(
            mm(["hello world", "hello"], padding=True)["input_ids"],
            [[2, 3], [2, 1]],
        )
        mm_padding = mm._tokenizer.padding
        text.encode("hello world")
        self.assertEqual(mm._tokenizer.padding, mm_padding)
        self.assertIsNone(text._tokenizer.padding)

        text("hello world hello", truncation=True, max_length=2)
        self.assertIsNotNone(text._tokenizer.truncation)
        self.assertIsNone(mm._tokenizer.truncation)
        self.assertEqual(mm.encode("hello world hello"), [2, 3, 2])

    def test_concurrent_text_and_padded_calls_keep_their_outputs(self):
        barrier = Barrier(2, timeout=10)
        prompt = "hello world " * 2000
        expected = [2, 3] * 2000

        def encode_text():
            for _ in range(30):
                barrier.wait()
                self.assertEqual(self.manager.tokenizer.encode(prompt), expected)

        def encode_multimodal_text():
            for _ in range(30):
                barrier.wait()
                self.assertEqual(
                    self.processor.tokenizer([prompt, "hello"], padding=True)[
                        "input_ids"
                    ],
                    [expected, [2] + [1] * (len(expected) - 1)],
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(encode_text),
                executor.submit(encode_multimodal_text),
            ]
            for future in futures:
                future.result(timeout=30)


if __name__ == "__main__":
    unittest.main()
