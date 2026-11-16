"""Offline regression checks: python -m unittest -v test_rag."""
import json
import unittest
from unittest.mock import MagicMock, patch

import rag


class RagTests(unittest.TestCase):
    def answer(self, result):
        sources = [{"id": "G03", "title": "Post Office", "text": "A grand hall.", "score": 0.8}]
        with patch.object(rag, "retrieve", return_value=sources), patch.object(
            rag, "_generate", return_value=json.dumps(result)
        ):
            return rag.answer_question("What can I see?", "http://test")

    def test_supported_answer_has_valid_citation(self):
        answer, _ = self.answer({"answer": "A grand hall.", "source_ids": ["G03"]})
        self.assertEqual(answer, "A grand hall. [G03]")

    def test_no_evidence_standardizes_refusal(self):
        answer, _ = self.answer({"answer": "The guide does not give ticket prices.", "source_ids": []})
        self.assertEqual(answer, rag.UNKNOWN)

    def test_unretrieved_citation_is_rejected(self):
        with self.assertRaises(rag.OllamaError):
            self.answer({"answer": "A grand hall.", "source_ids": ["G99"]})

    def test_empty_question_does_not_contact_server(self):
        with patch.object(rag, "retrieve") as retrieve:
            with self.assertRaises(ValueError):
                rag.answer_question("  ", "http://test")
            retrieve.assert_not_called()

    def test_stream_must_finish_and_not_hit_token_limit(self):
        for final in [None, {"done": True, "done_reason": "length"}]:
            with self.subTest(final=final):
                response = MagicMock()
                response.__enter__.return_value = response
                events = [{"message": {"content": '{"answer": "unfinished'}}]
                if final:
                    events.append(final)
                response.iter_lines.return_value = [json.dumps(e).encode() for e in events]
                with patch.object(rag.requests, "post", return_value=response):
                    with self.assertRaises(rag.OllamaError):
                        rag._generate("http://test", {})

    def test_stream_joins_complete_json(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [
            b'{"message":{"content":"hello "},"done":false}',
            b'{"message":{"content":"world"},"done":true,"done_reason":"stop"}',
        ]
        with patch.object(rag.requests, "post", return_value=response):
            self.assertEqual(rag._generate("http://test", {}), "hello world")


if __name__ == "__main__":
    unittest.main()
