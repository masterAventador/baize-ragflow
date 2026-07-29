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
import warnings
from types import SimpleNamespace

import pytest

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="\\[Errno 13\\] Permission denied\\.  joblib will operate in serial mode",
    category=UserWarning,
)

from api.db.services import document_service


class _FakeOrderField:
    def desc(self):
        return self

    def asc(self):
        return self


class _FakeField:
    def __eq__(self, other):
        return self

    def in_(self, other):
        # 返回标记，fake 查询的 where() 据此对 id 做真实过滤（延迟关联依赖它保真）
        return ("__id_in__", list(other))

    def not_in(self, other):
        return self


def _filter_id_in(rows, conds):
    """识别 _FakeField.in_ 产生的标记，按 id 做真实过滤（其余条件忽略，与旧行为一致）。"""
    for cond in conds:
        if isinstance(cond, tuple) and len(cond) == 2 and cond[0] == "__id_in__":
            allowed = set(cond[1])
            rows = [d for d in rows if d["id"] in allowed]
    return rows


class _FakeQuery:
    def __init__(self, docs):
        self._all = list(docs)
        self._current = list(docs)

    def join(self, *args, **kwargs):
        return self

    def where(self, *args, **kwargs):
        self._current = _filter_id_in(self._current, args)
        self._all = _filter_id_in(self._all, args)
        return self

    def __iter__(self):
        # 延迟关联第一步会迭代纯表查询取当页 id
        return iter([SimpleNamespace(**d) for d in self._current])

    def order_by(self, *args, **kwargs):
        return self

    def count(self):
        return len(self._all)

    def paginate(self, page, page_size):
        if page and page_size:
            start = (page - 1) * page_size
            end = start + page_size
            self._current = self._all[start:end]
        return self

    def dicts(self):
        return list(self._current)


@pytest.fixture
def metadata_calls(monkeypatch):
    sample_docs = [
        {"id": "doc-1"},
        {"id": "doc-2"},
        {"id": "doc-3"},
    ]

    model = SimpleNamespace(
        select=lambda *args, **kwargs: _FakeQuery(sample_docs),
        id=_FakeField(),
        kb_id=_FakeField(),
        name=_FakeField(),
        suffix=_FakeField(),
        run=_FakeField(),
        type=_FakeField(),
        created_by=_FakeField(),
        pipeline_id=_FakeField(),
        getter_by=lambda *_args, **_kwargs: _FakeOrderField(),
    )

    monkeypatch.setattr(document_service.DB, "connect", lambda *args, **kwargs: None)
    monkeypatch.setattr(document_service.DB, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(document_service.DocumentService, "model", model)
    monkeypatch.setattr(
        document_service.DocumentService,
        "get_cls_model_fields",
        classmethod(lambda cls: []),
    )

    calls = []

    def _fake_get_metadata_for_documents(cls, doc_ids, kb_id):
        calls.append((doc_ids, kb_id))
        return {doc_id: {"source_url": f"url-{doc_id}"} for doc_id in (doc_ids or [])}

    monkeypatch.setattr(
        document_service.DocMetadataService,
        "get_metadata_for_documents",
        classmethod(_fake_get_metadata_for_documents),
    )

    return calls


@pytest.mark.p2
def test_get_list_fetches_metadata_for_page_document_ids(metadata_calls):
    docs, count = document_service.DocumentService.get_list(
        "kb-1",
        1,
        2,
        "create_time",
        True,
        "",
        None,
        None,
    )

    assert count == 3
    assert [doc["id"] for doc in docs] == ["doc-1", "doc-2"]
    assert docs[0]["meta_fields"]["source_url"] == "url-doc-1"
    assert metadata_calls == [(["doc-1", "doc-2"], "kb-1")]


@pytest.mark.p2
def test_get_by_kb_id_fetches_metadata_for_page_document_ids(metadata_calls):
    docs, count = document_service.DocumentService.get_by_kb_id(
        "kb-1",
        2,
        1,
        "create_time",
        True,
        "",
        [],
        [],
        [],
        return_empty_metadata=False,
    )

    assert count == 3
    assert [doc["id"] for doc in docs] == ["doc-2"]
    assert docs[0]["meta_fields"]["source_url"] == "url-doc-2"
    assert metadata_calls == [(["doc-2"], "kb-1")]


@pytest.mark.p2
def test_get_by_kb_id_return_empty_metadata_keeps_dataset_wide_lookup(metadata_calls, monkeypatch):
    def _fake_get_metadata_for_documents(cls, doc_ids, kb_id):
        metadata_calls.append((doc_ids, kb_id))
        return {"doc-1": {"source_url": "url-doc-1"}} if doc_ids is None else {}

    monkeypatch.setattr(
        document_service.DocMetadataService,
        "get_metadata_for_documents",
        classmethod(_fake_get_metadata_for_documents),
    )

    docs, count = document_service.DocumentService.get_by_kb_id(
        "kb-1",
        1,
        2,
        "create_time",
        True,
        "",
        [],
        [],
        [],
        return_empty_metadata=True,
    )

    assert count == 3
    assert docs[0]["meta_fields"] == {}
    assert metadata_calls == [(None, "kb-1")]


# ── HugAI 性能补丁：count 不再走 4 表 JOIN，且可按需跳过 ──────────────────
#
# get_by_kb_id 原本用 `count = docs.count()` 对带 4 表 JOIN(File2Document/File/
# UserCanvas/User)的列表查询整包计数，76 万文档库上 3~4 秒、卡死单进程 API。JOIN 只
# 提供展示字段、无 WHERE 依赖，故 count 改用纯 Document 表的独立查询。

class _JoinTrackingQuery:
    """记录本查询是否链过 join / 是否被整表 paginate，用于断言查询形态。"""

    def __init__(self, docs, joined_flag_sink, paginated_join_sink=None):
        self._all = list(docs)
        self._current = list(docs)
        self._joined = False
        self._sink = joined_flag_sink
        self._paginated_join_sink = paginated_join_sink

    def join(self, *args, **kwargs):
        self._joined = True
        return self

    def where(self, *args, **kwargs):
        self._current = _filter_id_in(self._current, args)
        self._all = _filter_id_in(self._all, args)
        return self

    def order_by(self, *args, **kwargs):
        return self

    def paginate(self, page, page_size):
        if self._paginated_join_sink is not None:
            # 快照：paginate 落在带 join 的查询(旧行为=整表 JOIN 分页)还是纯表查询(延迟关联)
            self._paginated_join_sink.append(self._joined)
        if page and page_size:
            start = (page - 1) * page_size
            self._current = self._all[start:start + page_size]
        return self

    def count(self):
        # 快照：这次 count 是在链过 join 的查询上、还是独立查询上
        self._sink.append(self._joined)
        return len(self._all)

    def dicts(self):
        return list(self._current)

    def __iter__(self):
        # 延迟关联第一步会迭代纯表查询取当页 id
        return iter([SimpleNamespace(**d) if isinstance(d, dict) else d for d in self._current])


@pytest.fixture
def join_tracking(monkeypatch):
    """每次 select() 造一个新查询，共享一个 sink 记录每次 count 时的 join 状态。"""
    sample_docs = [{"id": "doc-1"}, {"id": "doc-2"}, {"id": "doc-3"}]
    count_join_flags = []

    def _select(*args, **kwargs):
        return _JoinTrackingQuery(sample_docs, count_join_flags)

    model = SimpleNamespace(
        select=_select,
        id=_FakeField(), kb_id=_FakeField(), name=_FakeField(), suffix=_FakeField(),
        run=_FakeField(), type=_FakeField(), created_by=_FakeField(), pipeline_id=_FakeField(),
        getter_by=lambda *_a, **_k: _FakeOrderField(),
    )
    monkeypatch.setattr(document_service.DB, "connect", lambda *a, **k: None)
    monkeypatch.setattr(document_service.DB, "close", lambda *a, **k: None)
    monkeypatch.setattr(document_service.DocumentService, "model", model)
    monkeypatch.setattr(document_service.DocumentService, "get_cls_model_fields", classmethod(lambda cls: []))
    monkeypatch.setattr(
        document_service.DocMetadataService, "get_metadata_for_documents",
        classmethod(lambda cls, doc_ids, kb_id: {}),
    )
    return count_join_flags


@pytest.mark.p2
def test_count_query_does_not_join(join_tracking):
    """count 必须走不带 join 的独立查询（这是本次性能修复的本质）。"""
    _docs, count = document_service.DocumentService.get_by_kb_id(
        "kb-1", 1, 2, "create_time", True, "", [], [], [],
    )
    assert count == 3
    assert join_tracking == [False]   # count 时未链过 join


@pytest.mark.p2
def test_need_count_false_skips_count(join_tracking):
    """need_count=False 时不计算 count（3 个不用 total 的调用点据此省掉开销）。"""
    _docs, count = document_service.DocumentService.get_by_kb_id(
        "kb-1", 0, 0, "create_time", True, "", [], [], [], need_count=False,
    )
    assert count == 0
    assert join_tracking == []        # 完全没调 count


@pytest.mark.p2
def test_suffix_has_default_value(join_tracking):
    """suffix 有默认值：修复 knowledgebase_service:125 少传 suffix 的 TypeError。"""
    # 只传到 types，不传 suffix —— 补丁前会 TypeError: missing 'suffix'
    _docs, count = document_service.DocumentService.get_by_kb_id(
        "kb-1", 1, 1000, "create_time", True, "", [], [],
    )
    assert count == 3


# ── HugAI 性能补丁(第二层)：分页走延迟关联，展示 JOIN 只作用于当页 id ──────────
#
# count 修掉后列表仍慢：带 4 表 JOIN 对 76 万行整表分页要 12s(JOIN 完再排序取 30 行)。
# 延迟关联 = 先纯 Document 表过滤+排序+分页拿当页 id(0.19s，kb_id/create_time 有索引)，
# 再对这 30 个 id 做 JOIN 取展示列(0.005s)。不分页调用(全量拉取)保持原查询。

@pytest.fixture
def paginate_tracking(monkeypatch):
    sample_docs = [{"id": "doc-1"}, {"id": "doc-2"}, {"id": "doc-3"}]
    count_join_flags = []
    paginate_join_flags = []

    def _select(*args, **kwargs):
        return _JoinTrackingQuery(sample_docs, count_join_flags, paginate_join_flags)

    model = SimpleNamespace(
        select=_select,
        id=_FakeField(), kb_id=_FakeField(), name=_FakeField(), suffix=_FakeField(),
        run=_FakeField(), type=_FakeField(), created_by=_FakeField(), pipeline_id=_FakeField(),
        getter_by=lambda *_a, **_k: _FakeOrderField(),
    )
    monkeypatch.setattr(document_service.DB, "connect", lambda *a, **k: None)
    monkeypatch.setattr(document_service.DB, "close", lambda *a, **k: None)
    monkeypatch.setattr(document_service.DocumentService, "model", model)
    monkeypatch.setattr(document_service.DocumentService, "get_cls_model_fields", classmethod(lambda cls: []))
    monkeypatch.setattr(
        document_service.DocMetadataService, "get_metadata_for_documents",
        classmethod(lambda cls, doc_ids, kb_id: {}),
    )
    return paginate_join_flags


@pytest.mark.p2
def test_paginated_list_uses_deferred_join(paginate_tracking):
    """分页时 paginate 必须落在纯表查询上（延迟关联），不允许对带 JOIN 的整表分页。"""
    docs, count = document_service.DocumentService.get_by_kb_id(
        "kb-1", 1, 2, "create_time", True, "", [], [], [],
    )
    assert count == 3
    assert len(docs) == 2                       # 分页语义不变
    assert paginate_tracking == [False]         # paginate 落在未 join 的纯表查询上


@pytest.mark.p2
def test_unpaginated_list_keeps_original_shape(paginate_tracking):
    """不分页(全量拉取)时不走延迟关联，保持原查询形态（不 paginate）。"""
    docs, _count = document_service.DocumentService.get_by_kb_id(
        "kb-1", 0, 0, "create_time", True, "", [], [], [], need_count=False,
    )
    assert len(docs) == 3                       # 全量返回
    assert paginate_tracking == []              # 全程没有 paginate
