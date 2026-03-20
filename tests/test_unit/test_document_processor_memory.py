"""
Memory optimization and regression tests for DocumentProcessor.

Tests verify:
1. _create_chunks does not infinite loop (critical regression)
2. PDF processing releases images properly
3. Memory stays bounded during large document processing
4. Image cleanup in _process_pdf, _process_docx, _process_pptx
"""

import gc
import io
import threading
import asyncio
import psutil
import os

import pytest
import fitz
from PIL import Image as PILImage
from unittest.mock import patch, MagicMock, AsyncMock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def doc_processor():
    """Create DocumentProcessor with mocked external services."""
    with patch("app.services.document_processor.get_vector_store") as mock_vs:
        mock_vs.return_value = MagicMock()
        mock_vs.return_value.add_documents = AsyncMock()

        from app.services.document_processor import DocumentProcessor

        processor = DocumentProcessor()
        processor.ocr_available = False
        processor.vision_available = False
        yield processor


def _get_rss_mb():
    """Get current RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _make_text_pdf(pages: int = 10) -> bytes:
    """Create a text-only PDF."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text(
            (72, 72), f"Page {i + 1}. " + "Lorem ipsum dolor sit amet. " * 50
        )
    data = doc.tobytes()
    doc.close()
    return data


def _make_image_pdf(
    pages: int = 20, images_per_page: int = 5, img_w: int = 800, img_h: int = 600
) -> bytes:
    """Create a PDF with embedded images."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} with images and content.")
        for j in range(images_per_page):
            img = PILImage.new("RGB", (img_w, img_h), (100, 150, 200))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            rect = fitz.Rect(72, 100 + j * 120, 400, 200 + j * 120)
            page.insert_image(rect, stream=buf.getvalue())
            img.close()
            del img
            buf.close()
            del buf
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------------------
# 1. _create_chunks infinite loop regression tests
# ---------------------------------------------------------------------------


class TestCreateChunksNoInfiniteLoop:
    """Ensure _create_chunks always terminates."""

    TIMEOUT_SECONDS = 5

    def _run_with_timeout(self, fn, timeout=None):
        timeout = timeout or self.TIMEOUT_SECONDS
        result = [None]
        error = [None]

        def target():
            try:
                result[0] = fn()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=target)
        t.start()
        t.join(timeout=timeout)
        assert not t.is_alive(), f"_create_chunks hung (infinite loop) after {timeout}s"
        if error[0]:
            raise error[0]
        return result[0]

    def test_short_text(self, doc_processor):
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks("Short text.", "doc1", "f.txt", 1)
        )
        assert len(chunks) == 1

    def test_text_slightly_over_chunk_size(self, doc_processor):
        """This was the exact trigger for the infinite loop."""
        text = "A" * 1001  # chunk_size=1000, so text is 1 char over
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1)
        )
        assert len(chunks) >= 2

    def test_text_no_separators_large(self, doc_processor):
        text = "A" * 5000
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1)
        )
        assert len(chunks) >= 5

    def test_normal_sentences(self, doc_processor):
        text = "Hello world. This is a test sentence. " * 100
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1)
        )
        assert len(chunks) >= 3

    def test_long_document(self, doc_processor):
        text = "The quick brown fox jumps over the lazy dog. " * 500
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1)
        )
        assert len(chunks) >= 20

    def test_very_long_document(self, doc_processor):
        text = (
            "Some repeated content with separators. "
            "Another sentence here! Yet more text? Indeed. "
        ) * 1000
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1),
            timeout=10,
        )
        assert len(chunks) >= 50

    def test_exactly_chunk_size(self, doc_processor):
        text = "A" * 1000
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1)
        )
        assert len(chunks) >= 1

    def test_empty_text(self, doc_processor):
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks("", "doc1", "f.txt", 1)
        )
        assert len(chunks) == 0

    def test_whitespace_only(self, doc_processor):
        chunks = self._run_with_timeout(
            lambda: doc_processor._create_chunks("   \n\n  \t  ", "doc1", "f.txt", 1)
        )
        assert len(chunks) == 0

    def test_overlap_equals_chunk_size(self, doc_processor):
        """Edge case: overlap >= chunk_size should not infinite loop."""
        original_overlap = doc_processor.chunk_overlap
        doc_processor.chunk_overlap = doc_processor.chunk_size  # pathological case
        try:
            text = "Hello world. " * 200
            chunks = self._run_with_timeout(
                lambda: doc_processor._create_chunks(text, "doc1", "f.txt", 1),
                timeout=10,
            )
            assert len(chunks) >= 1
        finally:
            doc_processor.chunk_overlap = original_overlap


# ---------------------------------------------------------------------------
# 2. Memory tests for document processing
# ---------------------------------------------------------------------------


class TestDocumentProcessingMemory:
    """Verify memory stays bounded during document processing."""

    @pytest.mark.asyncio
    async def test_text_file_memory(self, doc_processor):
        """Text file processing should use minimal memory."""
        gc.collect()
        before = _get_rss_mb()

        with (
            patch.object(doc_processor, "_store_chunks", new_callable=AsyncMock),
            patch.object(doc_processor, "_check_duplicate", return_value=None),
            patch.object(doc_processor, "_mark_processed"),
        ):
            r = await doc_processor.process_file(
                file_content=b"Hello world test content. " * 500,
                filename="test.txt",
                document_id="id1",
                metadata={},
            )

        gc.collect()
        after = _get_rss_mb()
        delta = after - before

        assert r.success
        assert len(r.chunks) > 0
        assert delta < 100, f"Text processing used {delta:.0f}MB (expected < 100MB)"

    @pytest.mark.asyncio
    async def test_text_pdf_memory(self, doc_processor):
        """Text-only PDF should use minimal memory."""
        pdf_bytes = _make_text_pdf(pages=10)
        gc.collect()
        before = _get_rss_mb()

        with (
            patch.object(doc_processor, "_store_chunks", new_callable=AsyncMock),
            patch.object(doc_processor, "_check_duplicate", return_value=None),
            patch.object(doc_processor, "_mark_processed"),
        ):
            r = await doc_processor.process_file(
                file_content=pdf_bytes,
                filename="text.pdf",
                document_id="id2",
                metadata={},
            )

        del pdf_bytes
        gc.collect()
        after = _get_rss_mb()
        delta = after - before

        assert r.success
        assert len(r.chunks) > 0
        assert delta < 100, f"Text PDF used {delta:.0f}MB (expected < 100MB)"

    @pytest.mark.asyncio
    async def test_image_pdf_memory(self, doc_processor):
        """PDF with 100 images should release memory after processing."""
        pdf_bytes = _make_image_pdf(pages=20, images_per_page=5)
        gc.collect()
        before = _get_rss_mb()

        with (
            patch.object(doc_processor, "_store_chunks", new_callable=AsyncMock),
            patch.object(doc_processor, "_check_duplicate", return_value=None),
            patch.object(doc_processor, "_mark_processed"),
        ):
            r = await doc_processor.process_file(
                file_content=pdf_bytes,
                filename="images.pdf",
                document_id="id3",
                metadata={},
            )

        del pdf_bytes
        gc.collect()
        after = _get_rss_mb()
        delta = after - before

        assert r.success
        assert delta < 300, f"Image PDF used {delta:.0f}MB (expected < 300MB)"

    @pytest.mark.asyncio
    async def test_huge_pdf_memory(self, doc_processor):
        """50-page PDF with 500 images should stay under 500MB."""
        pdf_bytes = _make_image_pdf(pages=50, images_per_page=10, img_w=1200, img_h=900)
        gc.collect()
        before = _get_rss_mb()

        with (
            patch.object(doc_processor, "_store_chunks", new_callable=AsyncMock),
            patch.object(doc_processor, "_check_duplicate", return_value=None),
            patch.object(doc_processor, "_mark_processed"),
        ):
            r = await doc_processor.process_file(
                file_content=pdf_bytes,
                filename="huge.pdf",
                document_id="id4",
                metadata={},
            )

        del pdf_bytes
        gc.collect()
        after = _get_rss_mb()
        delta = after - before

        assert r.success
        assert delta < 500, f"Huge PDF used {delta:.0f}MB (expected < 500MB)"

    @pytest.mark.asyncio
    async def test_file_path_mode_memory(self, doc_processor):
        """Processing via file_path should NOT load entire file into memory."""
        import tempfile

        pdf_bytes = _make_image_pdf(pages=10, images_per_page=3)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdf_bytes)
        tmp.close()
        del pdf_bytes

        gc.collect()
        before = _get_rss_mb()

        try:
            with (
                patch.object(doc_processor, "_store_chunks", new_callable=AsyncMock),
                patch.object(doc_processor, "_check_duplicate", return_value=None),
                patch.object(doc_processor, "_mark_processed"),
            ):
                r = await doc_processor.process_file(
                    filename="test_path.pdf",
                    document_id="id5",
                    metadata={},
                    file_path=tmp.name,
                )

            gc.collect()
            after = _get_rss_mb()
            delta = after - before

            assert r.success
            assert delta < 200, f"File-path mode used {delta:.0f}MB (expected < 200MB)"
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_sequential_processing_no_accumulation(self, doc_processor):
        """Processing multiple files sequentially should not accumulate memory."""
        gc.collect()
        baseline = _get_rss_mb()

        for i in range(5):
            pdf_bytes = _make_image_pdf(pages=10, images_per_page=3)
            with (
                patch.object(doc_processor, "_store_chunks", new_callable=AsyncMock),
                patch.object(doc_processor, "_check_duplicate", return_value=None),
                patch.object(doc_processor, "_mark_processed"),
            ):
                r = await doc_processor.process_file(
                    file_content=pdf_bytes,
                    filename=f"seq_{i}.pdf",
                    document_id=f"seq_{i}",
                    metadata={},
                )
            del pdf_bytes
            gc.collect()
            assert r.success

        gc.collect()
        final = _get_rss_mb()
        delta = final - baseline

        # After 5 files, memory should NOT grow linearly
        assert delta < 300, (
            f"Memory grew by {delta:.0f}MB after 5 files "
            f"(suggests accumulation/leak, expected < 300MB)"
        )
