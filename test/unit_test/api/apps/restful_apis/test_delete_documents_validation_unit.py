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
"""Unit tests for delete_documents id-ownership validation.

HugAI patch: the delete endpoint used to load EVERY document of the dataset
(O(dataset size)) just to check whether the requested ids belong to it — on a
760k-doc knowledge base that single ORM materialisation takes ~49s and wedges
the single-process ragflow API. These tests pin the new behaviour: validation
must query ONLY the requested ids (O(request size)) while producing the exact
same invalid_ids result as before.
"""

import importlib.util
from pathlib import Path

# 直接按文件路径加载 helper，绕过 api/apps/__init__.py（它 import quart，
# 单测环境未装）。helper 本身零外部依赖，故无需任何 stub。
_HELPER = Path(__file__).resolve().parents[5] / "api" / "apps" / "restful_apis" / "document_delete_helpers.py"
_spec = importlib.util.spec_from_file_location("document_delete_helpers", _HELPER)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
find_invalid_doc_ids = _mod.find_invalid_doc_ids


def test_returns_ids_not_belonging_to_dataset():
    """请求里不属于本库的 id 被识别为 invalid（语义与旧全表版一致）。"""
    # 本库实际拥有 d1 d2 d3；请求删 d2 d3 d9(不存在) dX(别的库)
    def query_existing(dataset_id, doc_ids):
        owned = {"d1", "d2", "d3"}
        # 定向查询语义：只返回「既在本库、又在请求列表里」的 id
        return {i for i in doc_ids if i in owned}

    invalid = find_invalid_doc_ids("kb-1", ["d2", "d3", "d9", "dX"], query_existing)
    assert invalid == ["d9", "dX"]


def test_all_valid_returns_empty():
    def query_existing(dataset_id, doc_ids):
        return set(doc_ids)  # 全都属于本库

    assert find_invalid_doc_ids("kb-1", ["d1", "d2"], query_existing) == []


def test_preserves_request_order_in_invalid_list():
    """invalid_ids 保持请求顺序（错误消息可复现，与旧实现一致）。"""
    def query_existing(dataset_id, doc_ids):
        return set()  # 一个都不属于

    assert find_invalid_doc_ids("kb-1", ["z", "a", "m"], query_existing) == ["z", "a", "m"]


def test_only_requested_ids_are_queried_not_whole_dataset():
    """核心：校验只把「请求的 id」下推给查询，绝不触发全库扫描。

    这是本次性能修复的本质——query_existing 收到的必须正好是请求列表，
    据此可用 `WHERE kb_id=? AND id IN (...)` 定向命中，而非拉全表。
    """
    seen = {}

    def query_existing(dataset_id, doc_ids):
        seen["dataset_id"] = dataset_id
        seen["doc_ids"] = list(doc_ids)
        return set(doc_ids)

    find_invalid_doc_ids("kb-7", ["a", "b", "c"], query_existing)
    assert seen["dataset_id"] == "kb-7"
    assert seen["doc_ids"] == ["a", "b", "c"]   # 只查这 3 个，不是整库
