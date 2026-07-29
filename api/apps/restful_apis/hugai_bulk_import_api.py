#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Temporary HugAI bulk import API extension.

RAGFlow auto-registers files under ``api/apps/restful_apis``. Keeping these
endpoints in an extension file avoids modifying upstream REST modules and makes
the short-lived HugAI initial-import branch easier to rebase or discard.
"""

import datetime
import logging
import time
from pathlib import Path

import xxhash

from api.apps import login_required
from api.apps.restful_apis.chunk_api import Chunk
from api.apps.restful_apis.hugai_folder_cache import TTLCache
from api.apps.services.document_api_service import map_doc_keys
from api.constants import FILE_NAME_LEN_LIMIT
from api.db import FileType
from api.db.joint_services.tenant_model_service import get_model_config_from_provider_instance
from api.db.services.doc_metadata_service import DocMetadataService
from api.db.services.document_service import DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.tenant_llm_service import TenantLLMService
from api.utils.api_utils import (
    add_tenant_id_to_kwargs,
    get_error_data_result,
    get_request_json,
    get_result,
    server_error_response,
)
from common import settings
from common.constants import LLMType, RetCode
from common.misc_utils import get_uuid, thread_pool_exec
from common.string_utils import is_content_empty
from common.tag_feature_utils import validate_tag_features

logger = logging.getLogger(__name__)

# 根/KB 文件夹查询缓存(见 _create_empty_profile_document 内注释)。TTL 5 分钟:
# 命中期内文件夹被手工删除/改名的病态场景最多陈旧 5 分钟即自愈。
_FOLDER_CACHE = TTLCache(ttl_seconds=300)


def _get_dataset_tenant_id(dataset_id):
    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return None
    return kb.tenant_id


def _normalize_chunk_requests(req):
    chunk_reqs = req.get("chunks")
    if not isinstance(chunk_reqs, list) or not chunk_reqs:
        return None, "`chunks` is required to be a non-empty list"
    normalized = []
    for idx, chunk_req in enumerate(chunk_reqs):
        if isinstance(chunk_req, str):
            chunk_req = {"content": chunk_req}
        if not isinstance(chunk_req, dict):
            return None, f"`chunks[{idx}]` is required to be an object"
        normalized.append(chunk_req)
    return normalized, None


def _build_manual_chunk(dataset_id, document_id, doc, req):
    """Build one text-only manual chunk document without embedding it."""
    from rag.nlp import rag_tokenizer

    if is_content_empty(req.get("content")):
        return None, "`content` is required"
    if "important_keywords" in req and not isinstance(req["important_keywords"], list):
        return None, "`important_keywords` is required to be a list"
    if "questions" in req and not isinstance(req["questions"], list):
        return None, "`questions` is required to be a list"

    important_keywords = [str(k).strip() for k in req.get("important_keywords", []) if str(k).strip()]
    questions = [str(q).strip() for q in req.get("questions", []) if str(q).strip()]
    content = req["content"]
    chunk_id = xxhash.xxh64((content + document_id).encode("utf-8")).hexdigest()
    d = {
        "id": chunk_id,
        "content_ltks": rag_tokenizer.tokenize(content),
        "content_with_weight": content,
    }
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    d["important_kwd"] = important_keywords
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(important_keywords))
    d["question_kwd"] = questions
    d["question_tks"] = rag_tokenizer.tokenize("\n".join(questions))
    d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
    d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
    d["kb_id"] = dataset_id
    d["docnm_kwd"] = doc.name
    d["doc_id"] = document_id

    if "tag_kwd" in req:
        if not isinstance(req["tag_kwd"], list):
            return None, "`tag_kwd` is required to be a list"
        if not all(isinstance(t, str) for t in req["tag_kwd"]):
            return None, "`tag_kwd` must be a list of strings"
        d["tag_kwd"] = req["tag_kwd"]
    if "tag_feas" in req:
        try:
            d["tag_feas"] = validate_tag_features(req["tag_feas"])
        except ValueError as exc:
            return None, f"`tag_feas` {exc}"

    return d, None


def _rename_chunk_for_response(d):
    key_mapping = {
        "id": "id",
        "content_with_weight": "content",
        "doc_id": "document_id",
        "important_kwd": "important_keywords",
        "tag_kwd": "tag_kwd",
        "question_kwd": "questions",
        "kb_id": "dataset_id",
        "create_timestamp_flt": "create_timestamp",
        "create_time": "create_time",
        "document_keyword": "document",
        "img_id": "image_id",
    }
    return {new_key: d[key] for key, new_key in key_mapping.items() if key in d}


def _embed_and_insert_manual_chunks(dataset_tenant_id, dataset_id, document_id, doc, chunks):
    """Embed and insert multiple manual chunks for one document in a single ES bulk."""
    from rag.nlp import search

    timings = {}
    embd_id = DocumentService.get_embd_id(document_id)
    model_config = get_model_config_from_provider_instance(dataset_tenant_id, LLMType.EMBEDDING.value, embd_id)
    embd_mdl = TenantLLMService.model_instance(model_config)

    texts = []
    for d in chunks:
        texts.append(doc.name)
        texts.append(d["content_with_weight"] if not d["question_kwd"] else "\n".join(d["question_kwd"]))
    start = time.perf_counter()
    vectors, token_count = embd_mdl.encode(texts)
    timings["embedding_seconds"] = time.perf_counter() - start

    for idx, d in enumerate(chunks):
        v = 0.1 * vectors[idx * 2] + 0.9 * vectors[idx * 2 + 1]
        d[f"q_{len(v)}_vec"] = v.tolist()

    start = time.perf_counter()
    insert_errors = settings.docStoreConn.insert(chunks, search.index_name(dataset_tenant_id), dataset_id)
    if insert_errors:
        raise RuntimeError(f"Failed to insert chunks: {insert_errors}")
    timings["chunk_es_insert_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    DocumentService.increment_chunk_num(doc.id, doc.kb_id, token_count, len(chunks), 0)
    timings["chunk_counter_seconds"] = time.perf_counter() - start
    return timings


def _find_existing_document(dataset_id, document_id, name):
    if document_id:
        docs = DocumentService.query(id=document_id, kb_id=dataset_id)
        if docs:
            return docs[0]
    docs = DocumentService.query(name=name, kb_id=dataset_id)
    return docs[0] if docs else None


def _create_empty_profile_document(dataset_id, kb, tenant_id, name):
    if not name:
        raise ValueError("File name can't be empty.")
    if len(name.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
        raise ValueError(f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.")
    if DocumentService.query(name=name, kb_id=dataset_id):
        raise ValueError("Duplicated document name in the same dataset.")

    timings = {}
    start = time.perf_counter()
    # get_kb_folder 内部的 get_root_folder 按 `parent_id = id` 过滤——两列自比较无索引可用，
    # 每次调用都对 file 表(生产 81.6 万行)全表扫描;首灌每建一个文档调一次，实测把 RDS CPU
    # 顶到 88%。根/KB 文件夹是 get-or-create 且建成后不变，按 tenant/kb 缓存(TTL 兜底)后
    # 全表扫描从「每文档一次」降为「每 TTL 窗口一次」。
    kb_root_folder = _FOLDER_CACHE.get_or_load(
        ("kb_root", kb.tenant_id),
        lambda: FileService.get_kb_folder(kb.tenant_id),
    )
    timings["document_get_root_folder_seconds"] = time.perf_counter() - start
    if not kb_root_folder:
        raise RuntimeError("Cannot find the root folder.")
    start = time.perf_counter()
    kb_folder = _FOLDER_CACHE.get_or_load(
        ("kb_folder", kb.tenant_id, kb.name),
        lambda: FileService.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"]),
    )
    timings["document_get_kb_folder_seconds"] = time.perf_counter() - start
    if not kb_folder:
        raise RuntimeError("Cannot find the kb folder for this file.")

    start = time.perf_counter()
    doc = DocumentService.insert(
        {
            "id": get_uuid(),
            "kb_id": kb.id,
            "parser_id": kb.parser_id,
            "pipeline_id": kb.pipeline_id,
            "parser_config": kb.parser_config,
            "created_by": tenant_id,
            "type": FileType.VIRTUAL,
            "name": name,
            "suffix": Path(name).suffix.lstrip("."),
            "location": "",
            "size": 0,
        }
    )
    timings["document_insert_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    FileService.add_file_from_kb(doc.to_dict(), kb_folder["id"], kb.tenant_id)
    timings["document_file_link_seconds"] = time.perf_counter() - start
    return doc, timings


def _reset_document_chunks(dataset_tenant_id, dataset_id, doc):
    """Clear existing chunks and reset document/KB counters before rewrite."""
    from rag.nlp import search

    index_name = search.index_name(dataset_tenant_id)
    if settings.docStoreConn.index_exist(index_name, dataset_id):
        settings.docStoreConn.delete({"doc_id": doc.id}, index_name, dataset_id)
    if (doc.token_num or 0) or (doc.chunk_num or 0):
        DocumentService.clear_chunk_num_when_rerun(doc.id)
    DocumentService.update_by_id(doc.id, {"token_num": 0, "chunk_num": 0, "process_duration": 0})


def _upsert_metadata_no_refresh(dataset_tenant_id, dataset_id, document_id, meta_fields):
    """Upsert metadata without per-document refresh in HugAI bulk-import mode."""
    start = time.perf_counter()
    processed_meta = DocMetadataService._split_combined_values(meta_fields or {})
    index_name = DocMetadataService._get_doc_meta_index_name(dataset_tenant_id)
    if not settings.docStoreConn.index_exist(index_name, ""):
        result = settings.docStoreConn.create_doc_meta_idx(index_name)
        if result is False:
            raise RuntimeError(f"Failed to create metadata index {index_name}")
    insert_errors = settings.docStoreConn.insert(
        [{"id": document_id, "kb_id": dataset_id, "meta_fields": processed_meta}],
        index_name,
        dataset_id,
    )
    if insert_errors:
        raise RuntimeError(f"Failed to upsert document metadata: {insert_errors}")
    return time.perf_counter() - start


def _upsert_hugai_profile_document_data(tenant_id, dataset_id, kb, name, meta_fields, chunk_reqs, req):
    """Synchronous heavy path for the HugAI profile upsert endpoint.

    The Quart route runs this in RAGFlow's shared thread pool so one blocking
    DashScope embedding call does not block the whole API event loop.
    """
    dataset_tenant_id = kb.tenant_id
    created_doc = None
    total_start = time.perf_counter()
    try:
        document_id = (req.get("document_id") or "").strip()
        start = time.perf_counter()
        doc = _find_existing_document(dataset_id, document_id, name)
        lookup_seconds = time.perf_counter() - start
        document_detail_timings = {}
        if doc is None:
            start = time.perf_counter()
            doc, document_detail_timings = _create_empty_profile_document(dataset_id, kb, tenant_id, name)
            created_doc = doc
            document_prepare_seconds = time.perf_counter() - start
        else:
            start = time.perf_counter()
            _reset_document_chunks(dataset_tenant_id, dataset_id, doc)
            document_prepare_seconds = time.perf_counter() - start

        start = time.perf_counter()
        chunks = []
        for idx, chunk_req in enumerate(chunk_reqs):
            d, err = _build_manual_chunk(dataset_id, doc.id, doc, chunk_req)
            if err:
                raise ValueError(f"`chunks[{idx}]`: {err}")
            chunks.append(d)
        build_chunk_seconds = time.perf_counter() - start

        timings = _embed_and_insert_manual_chunks(dataset_tenant_id, dataset_id, doc.id, doc, chunks)
        timings["metadata_seconds"] = _upsert_metadata_no_refresh(dataset_tenant_id, dataset_id, doc.id, meta_fields)
        timings["document_lookup_seconds"] = lookup_seconds
        timings["document_prepare_seconds"] = document_prepare_seconds
        timings.update(document_detail_timings)
        timings["chunk_build_seconds"] = build_chunk_seconds
        timings["total_seconds"] = time.perf_counter() - total_start
        renamed_chunks = [_rename_chunk_for_response(d) for d in chunks]
        for renamed_chunk in renamed_chunks:
            _ = Chunk(**renamed_chunk)
        return {"document": map_doc_keys(doc), "chunks": renamed_chunks, "timings": timings}
    except Exception:
        if created_doc is not None:
            try:
                DocumentService.remove_document(created_doc, dataset_tenant_id)
            except Exception:
                logger.warning("Failed to clean up HugAI bulk-import orphan document %s", created_doc.id)
        raise


def _add_chunks_bulk_data(dataset_tenant_id, dataset_id, document_id, doc, chunk_reqs):
    chunks = []
    for idx, chunk_req in enumerate(chunk_reqs):
        d, err = _build_manual_chunk(dataset_id, document_id, doc, chunk_req)
        if err:
            raise ValueError(f"`chunks[{idx}]`: {err}")
        chunks.append(d)

    timings = _embed_and_insert_manual_chunks(dataset_tenant_id, dataset_id, document_id, doc, chunks)
    renamed_chunks = [_rename_chunk_for_response(d) for d in chunks]
    for renamed_chunk in renamed_chunks:
        _ = Chunk(**renamed_chunk)
    return {"chunks": renamed_chunks, "timings": timings}


@manager.route("/datasets/<dataset_id>/hugai/profile-documents", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def upsert_hugai_profile_document(tenant_id, dataset_id):
    """Create or rewrite one HugAI profile document in one RAGFlow request."""
    if not KnowledgebaseService.accessible(kb_id=dataset_id, user_id=tenant_id):
        return get_error_data_result(message=f"You don't own the dataset {dataset_id}.")
    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return get_error_data_result(message=f"You don't own the dataset {dataset_id}.")

    req = await get_request_json()
    name = (req.get("name") or "").strip()
    if not name:
        return get_error_data_result(message="File name can't be empty.", code=RetCode.ARGUMENT_ERROR)
    if len(name.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
        return get_error_data_result(
            message=f"File name must be {FILE_NAME_LEN_LIMIT} bytes or less.",
            code=RetCode.ARGUMENT_ERROR,
        )
    meta_fields = req.get("meta_fields") or {}
    if not isinstance(meta_fields, dict):
        return get_error_data_result(message="`meta_fields` is required to be an object")
    chunk_reqs, err = _normalize_chunk_requests(req)
    if err:
        return get_error_data_result(message=err)

    try:
        data = await thread_pool_exec(
            _upsert_hugai_profile_document_data,
            tenant_id,
            dataset_id,
            kb,
            name,
            meta_fields,
            chunk_reqs,
            req,
        )
        return get_result(data=data)
    except ValueError as exc:
        return get_error_data_result(message=str(exc), code=RetCode.ARGUMENT_ERROR)
    except Exception as exc:
        return server_error_response(exc)


@manager.route("/datasets/<dataset_id>/documents/<document_id>/chunks/bulk", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def add_chunks_bulk(tenant_id, dataset_id, document_id):
    """Bulk manual chunk insert endpoint used by HugAI profile initial import."""
    if not KnowledgebaseService.accessible(kb_id=dataset_id, user_id=tenant_id):
        return get_error_data_result(message=f"You don't own the dataset {dataset_id}.")
    dataset_tenant_id = _get_dataset_tenant_id(dataset_id)
    if not dataset_tenant_id:
        return get_error_data_result(message=f"You don't own the dataset {dataset_id}.")
    doc = DocumentService.query(id=document_id, kb_id=dataset_id)
    if not doc:
        return get_error_data_result(message=f"You don't own the document {document_id}.")
    doc = doc[0]
    req = await get_request_json()
    chunk_reqs, err = _normalize_chunk_requests(req)
    if err:
        return get_error_data_result(message=err)

    try:
        data = await thread_pool_exec(_add_chunks_bulk_data, dataset_tenant_id, dataset_id, document_id, doc, chunk_reqs)
        return get_result(data=data)
    except ValueError as exc:
        return get_error_data_result(message=str(exc), code=RetCode.ARGUMENT_ERROR)
    except Exception as exc:
        return server_error_response(exc)
