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
"""Process-local TTL cache for tenant root / KB folder lookups (HugAI patch).

Profile bulk-import resolves the tenant root folder for EVERY document it
creates. ``FileService.get_root_folder`` filters ``parent_id = id`` — a
two-column self-comparison no ordinary index can serve — so each call full-scans
the ``file`` table (816k rows in production, measured to monopolise RDS CPU at
~88% during backfill, several scans per second). Root and KB folders are
get-or-create rows that never change once created, so a short-TTL cache removes
those scans entirely: one scan per TTL window instead of one per document.

Kept dependency-free (stdlib only) so it can be unit-tested without booting the
web app. TTL is a safety net for the pathological cases (folder manually
deleted / renamed): a stale entry self-heals within ``ttl_seconds``.
"""

import threading
import time


class TTLCache:
    """Tiny thread-safe get-or-load cache with per-entry TTL.

    ``None``/falsy loader results are NOT cached — "folder does not exist yet"
    must stay a transient state, otherwise a pre-creation lookup would pin the
    miss until the TTL expires.

    Cache hits return the SAME object reference that was stored — callers must
    treat returned values as read-only; mutating them in place would poison the
    cache for every subsequent caller.
    """

    def __init__(self, ttl_seconds=300, now_fn=time.monotonic):
        self._ttl = ttl_seconds
        self._now = now_fn
        self._lock = threading.Lock()
        self._entries = {}  # key -> (expires_at, value)

    def get_or_load(self, key, loader):
        now = self._now()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] > now:
                return entry[1]
        # loader 在锁外执行：它可能是一次数据库查询，持锁会把并发请求串行在慢查询后面。
        # 并发首次加载时可能重复执行 loader——loader 均为幂等 get-or-create，重复无害。
        value = loader()
        if value:
            with self._lock:
                self._entries[key] = (now + self._ttl, value)
        return value

    def invalidate(self, key=None):
        with self._lock:
            if key is None:
                self._entries.clear()
            else:
                self._entries.pop(key, None)
