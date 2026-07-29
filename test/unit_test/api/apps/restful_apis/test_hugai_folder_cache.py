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
"""Unit tests for the HugAI folder TTL cache.

HugAI patch: profile bulk-import resolves the tenant root folder for EVERY
document it creates. `FileService.get_root_folder` filters `parent_id = id`
(self-reference) — no ordinary index can serve a two-column comparison, so on
an 816k-row `file` table each call is a full scan (~monopolises RDS CPU during
backfill, measured 88%). Root/KB folders are get-or-create and effectively
immutable, so a small TTL cache removes the scans entirely.
"""

import importlib.util
from pathlib import Path

# 零依赖模块,按文件路径直接加载,绕过 api/apps/__init__ 的 quart 依赖。
_MOD_PATH = Path(__file__).resolve().parents[5] / "api" / "apps" / "restful_apis" / "hugai_folder_cache.py"
_spec = importlib.util.spec_from_file_location("hugai_folder_cache", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TTLCache = _mod.TTLCache


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_first_call_loads_and_returns_value():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=300, now_fn=clock)
    calls = []

    value = cache.get_or_load("k1", lambda: calls.append(1) or {"id": "root-1"})
    assert value == {"id": "root-1"}
    assert calls == [1]


def test_second_call_hits_cache_without_loader():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=300, now_fn=clock)
    calls = []

    cache.get_or_load("k1", lambda: calls.append(1) or {"id": "root-1"})
    value = cache.get_or_load("k1", lambda: calls.append(2) or {"id": "other"})
    assert value == {"id": "root-1"}     # 命中缓存,loader 不再执行
    assert calls == [1]


def test_expired_entry_reloads():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=300, now_fn=clock)
    calls = []

    cache.get_or_load("k1", lambda: calls.append(1) or {"id": "v1"})
    clock.now += 301                      # 越过 TTL
    value = cache.get_or_load("k1", lambda: calls.append(2) or {"id": "v2"})
    assert value == {"id": "v2"}
    assert calls == [1, 2]


def test_none_result_is_not_cached():
    """loader 返回 None(如根目录尚未建成)不得缓存,否则会把「暂时不存在」钉死。"""
    clock = _Clock()
    cache = TTLCache(ttl_seconds=300, now_fn=clock)
    calls = []

    assert cache.get_or_load("k1", lambda: calls.append(1)) is None   # append 返回 None
    value = cache.get_or_load("k1", lambda: calls.append(2) or {"id": "late"})
    assert value == {"id": "late"}        # 第二次仍执行 loader
    assert calls == [1, 2]


def test_keys_are_independent():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=300, now_fn=clock)

    a = cache.get_or_load(("kb_root", "t1"), lambda: {"id": "r1"})
    b = cache.get_or_load(("kb_root", "t2"), lambda: {"id": "r2"})
    assert a["id"] == "r1" and b["id"] == "r2"


def test_invalidate_forces_reload():
    clock = _Clock()
    cache = TTLCache(ttl_seconds=300, now_fn=clock)
    calls = []

    cache.get_or_load("k1", lambda: calls.append(1) or {"id": "v1"})
    cache.invalidate("k1")
    cache.get_or_load("k1", lambda: calls.append(2) or {"id": "v2"})
    assert calls == [1, 2]
