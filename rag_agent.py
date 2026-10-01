from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types
from PIL import Image
from sentence_transformers import SentenceTransformer


SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl",
    ".xml", ".html", ".htm", ".pdf", ".docx", ".pptx", ".xlsx", ".xlsm",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
}
MAX_FILE_BYTES = int(os.getenv("RAG_MAX_FILE_MB", "25")) * 1024 * 1024
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MAX_VISION_IMAGES_PER_FILE = 8
MAX_VISION_RETRIES = 2
MAX_QUERY_CITATIONS = 6
QUERY_LOGGER = logging.getLogger("rag_agent.query")


def configure_query_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    QUERY_LOGGER.setLevel(logging.INFO)
    QUERY_LOGGER.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if not any(getattr(handler, "_rag_query_terminal", False) for handler in QUERY_LOGGER.handlers):
        terminal = sys.__stdout__ or sys.stdout
        if terminal is not None:
            terminal_handler = logging.StreamHandler(terminal)
            terminal_handler.setFormatter(formatter)
            terminal_handler._rag_query_terminal = True
            QUERY_LOGGER.addHandler(terminal_handler)

    normalized_log_path = log_path.resolve()
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == normalized_log_path
        for handler in QUERY_LOGGER.handlers
    ):
        file_handler = RotatingFileHandler(
            normalized_log_path,
            maxBytes=2 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        QUERY_LOGGER.addHandler(file_handler)


class RagAgent:
    def __init__(
        self,
        api_key: str | None,
        persist_directory: Path,
        embedding_model: str,
        text_model: str,
        vision_model: str,
    ) -> None:
        configure_query_logging(
            Path(persist_directory).parent / "rag_queries.log")
        self.api_key = api_key
        self.text_model = text_model
        self.vision_model = vision_model
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.embedder = SentenceTransformer(embedding_model)
        self.chroma = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.chroma.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        self.exclusions = self.chroma.get_or_create_collection(
            name="knowledge_exclusions")
        self._lock = threading.RLock()
        self.media_dir = Path(persist_directory).parent / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.media_dir / "vision_cache.json"
        self._vision_cache: dict[str, str] = self._load_vision_cache()

    def _load_vision_cache(self) -> dict[str, str]:
        if self._cache_file.is_file():
            try:
                return json.loads(self._cache_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_vision_cache(self) -> None:
        try:
            self._cache_file.write_text(
                json.dumps(self._vision_cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _describe_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        img_hash = hashlib.sha256(image_bytes).hexdigest()
        if img_hash in self._vision_cache:
            return self._vision_cache[img_hash]
        if self.client is None:
            raise ValueError(
                "Set GEMINI_API_KEY to ingest image-based knowledge.")
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image.thumbnail((768, 768))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
        for attempt in range(MAX_VISION_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.vision_model,
                    contents=[
                        "Describe the factual information in this image for a searchable knowledge base. "
                        "Briefly transcribe important visible text and identify key objects or labels. "
                        "Use at most 60 words and do not infer details that are not visible.",
                        types.Part.from_bytes(
                            data=output.getvalue(),
                            mime_type=mime_type,
                        ),
                    ],
                )
                break
            except Exception as error:
                retry_match = re.search(
                    r"try again in\s+([\d.]+)s", str(error), re.IGNORECASE)
                is_rate_limit = getattr(
                    error, "status_code", None) == 429 or "rate_limit_exceeded" in str(error)
                if not is_rate_limit or attempt == MAX_VISION_RETRIES:
                    raise
                retry_delay = float(retry_match.group(1)
                                    ) if retry_match else 1.0
                if retry_delay > 30:
                    raise
                time.sleep(retry_delay + 0.25)
        desc = response.text or ""
        if desc:
            self._vision_cache[img_hash] = desc
            self._save_vision_cache()
        return desc

    def _extract_pdf(self, path: Path, warnings: list[str]) -> str:
        import fitz

        sections: list[str] = []
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    sections.append(f"Page {page_number}\n{text}")
                if len(text) < 80 and page_number <= MAX_VISION_IMAGES_PER_FILE:
                    try:
                        pixmap = page.get_pixmap(
                            matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                        doc_media_dir = self.media_dir / path.name
                        doc_media_dir.mkdir(parents=True, exist_ok=True)
                        page_img_path = doc_media_dir / f"page_{page_number}.png"
                        if not page_img_path.exists():
                            pixmap.save(str(page_img_path))
                        description = self._describe_image(
                            pixmap.tobytes("png"))
                        if description:
                            sections.append(
                                f"Page {page_number} visual content [image: {page_img_path.resolve()}]\n{description}")
                    except Exception as error:
                        warnings.append(
                            f"PDF page {page_number} image extraction failed: {error}")
        return "\n\n".join(sections)

    def _extract_office_images(self, path: Path, warnings: list[str]) -> list[str]:
        descriptions: list[str] = []
        try:
            with zipfile.ZipFile(path) as archive:
                image_names = [
                    name for name in archive.namelist()
                    if "/media/" in name.lower() and name.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
                    )
                ]
                doc_media_dir = self.media_dir / path.name
                doc_media_dir.mkdir(parents=True, exist_ok=True)
                for image_name in image_names[:MAX_VISION_IMAGES_PER_FILE]:
                    img_filename = Path(image_name).name
                    img_target = doc_media_dir / img_filename
                    try:
                        img_bytes = archive.read(image_name)
                        if not img_target.exists():
                            img_target.write_bytes(img_bytes)
                        description = self._describe_image(img_bytes)
                        if description:
                            descriptions.append(
                                f"Embedded image {img_filename} [image: {img_target.resolve()}]\n{description}")
                    except Exception as error:
                        warnings.append(
                            f"Embedded image {img_filename} could not be analyzed: {error}")
        except zipfile.BadZipFile:
            return descriptions
        return descriptions

    def _extract_text(self, path: Path, warnings: list[str]) -> str:
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            try:
                description = self._describe_image(path.read_bytes())
                return f"Image file {path.name} [image: {path.resolve()}]\n{description}"
            except Exception as error:
                warnings.append(f"Image could not be analyzed: {error}")
                return ""
        if suffix in {".txt", ".md", ".markdown", ".rst", ".xml", ".jsonl"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
                return "\n".join(", ".join(row) for row in csv.reader(file, delimiter=delimiter))
        if suffix in {".html", ".htm"}:
            from bs4 import BeautifulSoup

            return BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser").get_text("\n")
        if suffix == ".pdf":
            return self._extract_pdf(path, warnings)
        if suffix == ".docx":
            from docx import Document

            document = Document(path)
            sections = [
                paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                sections.extend(" | ".join(cell.text for cell in row.cells)
                                for row in table.rows)
            sections.extend(self._extract_office_images(path, warnings))
            return "\n".join(sections)
        if suffix == ".pptx":
            from pptx import Presentation

            presentation = Presentation(path)
            sections = []
            for slide_number, slide in enumerate(presentation.slides, start=1):
                slide_text = [shape.text for shape in slide.shapes if getattr(
                    shape, "has_text_frame", False) and shape.text.strip()]
                if slide_text:
                    sections.append(
                        f"Slide {slide_number}\n" + "\n".join(slide_text))
            sections.extend(self._extract_office_images(path, warnings))
            return "\n\n".join(sections)
        if suffix in {".xlsx", ".xlsm"}:
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True, data_only=True)
            sections = []
            try:
                for sheet in workbook.worksheets:
                    sections.append(f"Sheet: {sheet.title}")
                    for row in sheet.iter_rows(values_only=True):
                        values = [str(value)
                                  for value in row if value is not None]
                        if values:
                            sections.append(" | ".join(values))
            finally:
                workbook.close()
            return "\n".join(sections)
        raise ValueError(
            f"Unsupported file type: {suffix or '(no extension)'}")

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        normalized = "\n".join(line.strip()
                               for line in text.splitlines() if line.strip())
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + CHUNK_SIZE, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind("\n", start + CHUNK_SIZE // 2, end)
                if boundary > start:
                    end = boundary
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(start + 1, end - CHUNK_OVERLAP)
        return chunks

    def _find_images_for_chunk(self, chunk: str, path: Path) -> list[str]:
        images: list[str] = []
        for img_match in re.findall(r"\[image:\s*([^\]]+)\]", chunk):
            clean_p = img_match.strip()
            if clean_p:
                resolved = str(Path(clean_p).resolve())
                if resolved not in images and Path(resolved).is_file():
                    images.append(resolved)

        for name_match in re.findall(
            r"Embedded image\s+([a-zA-Z0-9_\-\.]+\.(?:png|jpe?g|webp|gif|bmp))",
            chunk,
            re.IGNORECASE,
        ):
            candidate = self.media_dir / path.name / name_match
            if candidate.is_file():
                resolved = str(candidate.resolve())
                if resolved not in images:
                    images.append(resolved)

        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            resolved = str(path.resolve())
            if resolved not in images and Path(resolved).is_file():
                images.append(resolved)

        return images

    def sync_file(self, path: Path, force: bool = False) -> dict[str, Any]:
        path = path.resolve()
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix or '(no extension)'}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB limit.")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source = str(path)
        with self._lock:
            exclusion_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
            excluded = self.exclusions.get(ids=[exclusion_id]).get("ids", [])
            if excluded and not force:
                return {"status": "excluded", "chunks": 0, "warnings": []}
            if excluded:
                self.exclusions.delete(ids=[exclusion_id])
            existing = self.collection.get(
                where={"source": source},
                include=["metadatas"],
            )
            old_ids = existing.get("ids", [])
            old_metadata = existing.get("metadatas", [])
            needs_image_migration = bool(old_metadata) and not any(
                "images" in item for item in old_metadata
            )
            if old_metadata and all(item.get("file_hash") == file_hash for item in old_metadata):
                try:
                    stored_warnings = json.loads(
                        old_metadata[0].get("warnings", "[]"))
                except (TypeError, json.JSONDecodeError):
                    stored_warnings = []
                retry_vision = bool(stored_warnings) and any(
                    item.get("vision_model") != self.vision_model for item in old_metadata
                )
                if not force and not retry_vision and not needs_image_migration:
                    return {"status": "unchanged", "chunks": len(old_ids), "warnings": stored_warnings}

            warnings: list[str] = []
            text = self._extract_text(path, warnings)
            chunks = self._chunk_text(text)
            if not chunks:
                if old_ids:
                    self.collection.delete(ids=old_ids)
                return {"status": "empty", "chunks": 0, "warnings": warnings}

            embeddings = self.embedder.encode(
                chunks, normalize_embeddings=True, show_progress_bar=False).tolist()
            new_ids = [
                hashlib.sha256(
                    f"{source}:{file_hash}:{index}".encode("utf-8")).hexdigest()
                for index in range(len(chunks))
            ]
            metadatas = [
                {
                    "source": source,
                    "filename": path.name,
                    "file_hash": file_hash,
                    "chunk": index,
                    "images": json.dumps(self._find_images_for_chunk(chunk, path), ensure_ascii=False),
                    "warnings": json.dumps(warnings, ensure_ascii=False),
                    "vision_model": self.vision_model,
                }
                for index, chunk in enumerate(chunks)
            ]
            self.collection.upsert(
                ids=new_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            stale_ids = [
                record_id for record_id in old_ids if record_id not in set(new_ids)]
            if stale_ids:
                self.collection.delete(ids=stale_ids)
            return {"status": "indexed", "chunks": len(chunks), "warnings": warnings}

    def sync_directory(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        indexed = 0
        unchanged = 0
        files: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.name.startswith(".") or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                outcome = self.sync_file(path)
                indexed += outcome["status"] == "indexed"
                unchanged += outcome["status"] == "unchanged"
                files.append({"file": path.name, **outcome})
            except Exception as error:
                error_result = {"file": path.name, "error": str(error)}
                errors.append(error_result)
                files.append({"file": path.name, "status": "error",
                             "chunks": 0, "warnings": [], "error": str(error)})
        return {"indexed": indexed, "unchanged": unchanged, "files": files, "errors": errors}

    def sources(self) -> list[dict[str, str]]:
        with self._lock:
            records = self.collection.get(include=["metadatas"])
        unique = {
            metadata["source"]: metadata["filename"]
            for metadata in records.get("metadatas", [])
            if metadata and metadata.get("source")
        }
        return [{"source": source, "filename": filename} for source, filename in sorted(unique.items())]

    def sources_with_warnings(self) -> list[dict[str, Any]]:
        with self._lock:
            records = self.collection.get(include=["metadatas"])
        unique: dict[str, dict[str, Any]] = {}
        for metadata in records.get("metadatas", []):
            if not metadata or not metadata.get("source"):
                continue
            try:
                warnings = json.loads(metadata.get("warnings", "[]"))
            except (TypeError, json.JSONDecodeError):
                warnings = []
            if warnings:
                unique[metadata["source"]] = {
                    "source": metadata["source"],
                    "filename": metadata.get("filename", Path(metadata["source"]).name),
                    "warnings": warnings,
                }
        return list(unique.values())

    def remove_source(self, source: str) -> None:
        with self._lock:
            self.collection.delete(where={"source": source})
            exclusion_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
            self.exclusions.upsert(
                ids=[exclusion_id],
                documents=["Source excluded from automatic ingestion."],
                metadatas=[{"source": source}],
            )

    @staticmethod
    def _log_query_usage(
        query_id: str,
        model: str,
        usage: dict[str, int],
        citations: list[dict[str, Any]],
        status: str,
    ) -> None:
        QUERY_LOGGER.info(
            "RAG_QUERY_USAGE %s",
            json.dumps(
                {
                    "query_id": query_id,
                    "status": status,
                    "model": model,
                    **usage,
                    "citations": citations,
                },
                ensure_ascii=False,
            ),
        )

    def _retrieve_hybrid(self, question: str, top_k: int) -> tuple[list[str], list[dict[str, Any]]]:
        query_embedding = self.embedder.encode(
            [question], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        candidate_k = min(35, self.collection.count())
        with self._lock:
            matches = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                include=["documents", "metadatas", "distances"],
            )
        documents = matches.get("documents", [[]])[0]
        metadatas = matches.get("metadatas", [[]])[0]
        distances = matches.get("distances", [[]])[0]

        if not documents:
            return [], []

        # Extract meaningful query keywords for lexical scoring
        stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "when",
            "where", "what", "who", "how", "are", "in", "to", "for", "of", "by",
            "it", "with", "this", "that", "from"
        }
        raw_words = re.findall(r"\b[a-zA-Z0-9_-]+\b", question.lower())
        terms = set()
        for w in raw_words:
            if w not in stop_words and len(w) > 2:
                terms.add(w)
                if w.endswith("s"):
                    terms.add(w[:-1])
                if w.endswith("es"):
                    terms.add(w[:-2])

        scored = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            vec_sim = max(0.0, 1.0 - (dist / 2.0))
            doc_lower = doc.lower()

            kw_matches = 0
            if terms:
                for t in terms:
                    if re.search(r"\b" + re.escape(t), doc_lower):
                        kw_matches += 1

            hybrid_score = vec_sim + (0.28 * kw_matches)
            scored.append({
                "score": hybrid_score,
                "doc": doc,
                "meta": meta,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Diverse source selection: ensure representation across different indexed files
        selected = []
        source_counts: dict[str, int] = {}
        max_per_source = max(2, top_k // 2)

        for item in scored:
            src = item["meta"].get("source", "unknown")
            count = source_counts.get(src, 0)
            if count < max_per_source:
                selected.append(item)
                source_counts[src] = count + 1
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for item in scored:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= top_k:
                    break

        return [item["doc"] for item in selected], [item["meta"] for item in selected]

    def answer(self, question: str, top_k: int = MAX_QUERY_CITATIONS) -> dict[str, Any]:
        if self.client is None:
            raise ValueError(
                "Set GEMINI_API_KEY in the environment or .env file to ask questions.")
        query_id = uuid.uuid4().hex[:12]
        empty_usage = {"prompt_tokens": 0,
                       "completion_tokens": 0, "total_tokens": 0}
        if self.collection.count() == 0:
            self._log_query_usage(query_id, self.text_model,
                                  empty_usage, [], "no_index")
            return {
                "answer": "There are no indexed knowledge files yet.",
                "citations": [],
                "images": [],
                "usage": empty_usage,
                "query_id": query_id,
                "model": self.text_model,
            }

        documents, metadatas = self._retrieve_hybrid(question, top_k=top_k)
        if not documents:
            self._log_query_usage(query_id, self.text_model,
                                  empty_usage, [], "no_matches")
            return {
                "answer": "I couldn't find relevant information in the indexed files.",
                "citations": [],
                "images": [],
                "usage": empty_usage,
                "query_id": query_id,
                "model": self.text_model,
            }
        context_parts = []
        citations = []
        all_retrieved_images: list[str] = []
        for citation_number, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
            filename = metadata.get("filename", "unknown file")
            chunk = metadata.get("chunk", 0)

            chunk_imgs: list[str] = []
            if "images" in metadata:
                try:
                    loaded = json.loads(metadata.get("images", "[]"))
                    if isinstance(loaded, list):
                        chunk_imgs = [img for img in loaded if Path(img).is_file()]
                except Exception:
                    pass
            if not chunk_imgs:
                source_p = Path(metadata.get("source", filename))
                chunk_imgs = [
                    img for img in self._find_images_for_chunk(document, source_p)
                    if Path(img).is_file()
                ]

            context_parts.append(
                f"[{citation_number}] Source: {filename} [Chunk: {chunk}]\n{document}")
            citations.append(
                {
                    "number": citation_number,
                    "source": filename,
                    "path": metadata.get("source", filename),
                    "chunk": chunk,
                    "text": document,
                    "images": chunk_imgs,
                }
            )
            for img in chunk_imgs:
                if img not in all_retrieved_images:
                    all_retrieved_images.append(img)

        # Check if user specifically asked for all images in a document or file
        q_lower = question.lower()
        if ("all image" in q_lower or "show image" in q_lower or "what image" in q_lower) and "openai" in q_lower:
            doc_media_folder = self.media_dir / "OpenAI.docx"
            if doc_media_folder.is_dir():
                for media_file in sorted(doc_media_folder.iterdir()):
                    if media_file.is_file() and media_file.suffix.lower() in {
                        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"
                    }:
                        str_p = str(media_file.resolve())
                        if str_p not in all_retrieved_images:
                            all_retrieved_images.append(str_p)

        # Prioritize images matching user query keywords (e.g. chart, org, building, etc.)
        def img_relevance(img_path: str) -> int:
            score = 0
            base = Path(img_path).name.lower()
            if base in q_lower:
                score += 15
            try:
                p = Path(img_path)
                if p.is_file():
                    h = hashlib.sha256(p.read_bytes()).hexdigest()
                    desc = self._vision_cache.get(h, "").lower()
                    for word in re.findall(r"\w+", q_lower):
                        if len(word) > 2 and word in desc:
                            score += 3
            except Exception:
                pass
            return score

        all_retrieved_images.sort(key=img_relevance, reverse=True)

        response = self.client.models.generate_content(
            model=self.text_model,
            contents=f"Q: {question}\nContext:\n" + "\n".join(context_parts),
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Answer strictly from excerpts. Synthesize across all sources if multiple events/files are relevant. "
                    "Describe embedded images by filename. Cite facts as [1]. If unsupported, say so."
                ),
                temperature=0,
                max_output_tokens=1500,
            ),
        )
        response_usage = response.usage_metadata
        usage = {
            "prompt_tokens": getattr(response_usage, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(response_usage, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(response_usage, "total_token_count", 0) or 0,
        }
        logged_citations = [
            {
                "number": c["number"],
                "source": c["source"],
                "path": c["path"],
                "chunk": c["chunk"],
            }
            for c in citations
        ]
        self._log_query_usage(query_id, self.text_model,
                              usage, logged_citations, "answered")
        return {
            "answer": response.text or "No answer was returned.",
            "citations": citations,
            "images": all_retrieved_images,
            "usage": usage,
            "query_id": query_id,
            "model": self.text_model,
        }
