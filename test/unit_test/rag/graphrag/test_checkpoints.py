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

import pytest
from valkey.cluster import key_slot

from rag.graphrag import checkpoints


class _FakeRedisClient:
    def __init__(self, conn):
        self.conn = conn
        self.expirations = {}
        self.scan_counts = []

    def expire(self, key, ttl):
        self.expirations[key] = ttl
        return True

    def pipeline(self, transaction=True):
        raise AssertionError("Cluster checkpoint code must not use transaction pipelines")

    def sscan_iter(self, key, count=None):
        self.scan_counts.append((key, count))
        yield from self.conn.sets.get(key, set())


class _FakeRedisConn:
    def __init__(self):
        self.values = {}
        self.sets = {}
        self.REDIS = _FakeRedisClient(self)
        self.fail_set = False
        self.fail_atomic_save = False

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, exp=3600):
        if self.fail_set:
            return False
        self.values[key] = value
        self.REDIS.expire(key, exp)
        return True

    def sadd(self, key, member):
        self.sets.setdefault(key, set()).add(member)
        return True

    def smembers(self, key):
        raise AssertionError("checkpoint code must use sscan_iter instead of smembers")

    def sscan_iter(self, key, *, count=None):
        return self.REDIS.sscan_iter(key, count=count)

    def save_indexed_value(self, *, data_key, value, index_key, member, ttl):
        if self.fail_atomic_save:
            raise RuntimeError("redis atomic save failed")
        self.values[data_key] = value
        self.REDIS.expire(data_key, ttl)
        self.sets.setdefault(index_key, set()).add(member)
        self.REDIS.expire(index_key, ttl)
        return True

    def delete(self, key):
        self.values.pop(key, None)
        self.sets.pop(key, None)
        return True


@pytest.fixture
def fake_redis(monkeypatch):
    fake = _FakeRedisConn()
    monkeypatch.setattr(checkpoints, "REDIS_CONN", fake)
    return fake


@pytest.mark.p1
def test_checkpoint_keys_are_stable():
    first = checkpoints.community_checkpoint_key("1", "2", ["B", "A"])
    second = checkpoints.community_checkpoint_key("1", "2", ["A", "B"])
    assert first == second

    pairs = [("alpha", "alfa"), ("beta", "bata")]
    reversed_pairs = list(reversed(pairs))
    assert checkpoints.resolution_checkpoint_key("entity", pairs) == checkpoints.resolution_checkpoint_key("entity", reversed_pairs)

    internally_reversed_pairs = [("alfa", "alpha"), ("bata", "beta")]
    assert checkpoints.resolution_checkpoint_key("entity", pairs) == checkpoints.resolution_checkpoint_key("entity", internally_reversed_pairs)


@pytest.mark.p1
def test_checkpoint_index_and_data_keys_share_cluster_slot():
    index_key = checkpoints._checkpoint_index_key("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT)
    data_key = checkpoints._checkpoint_data_key("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT, "key-1")

    assert key_slot(index_key.encode()) == key_slot(data_key.encode())


@pytest.mark.p1
@pytest.mark.asyncio
async def test_load_checkpoints_reads_redis_index(fake_redis, monkeypatch):
    await checkpoints.save_checkpoint("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT, "k1", {"value": 1})
    await checkpoints.save_checkpoint("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT, "k2", {"value": 2})
    await checkpoints.save_checkpoint("tenant-1", "kb-2", checkpoints.COMMUNITY_CHECKPOINT, "k3", {"value": 3})

    thread_pool_calls = []

    async def _fake_thread_pool_exec(func, *args, **kwargs):
        thread_pool_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(checkpoints, "thread_pool_exec", _fake_thread_pool_exec)

    loaded = await checkpoints.load_checkpoints("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT, page_size=1)

    assert loaded == {"k1": {"value": 1}, "k2": {"value": 2}}
    assert thread_pool_calls == [
        (
            checkpoints._load_checkpoints_sync,
            ("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT, 1),
            {},
        )
    ]
    assert fake_redis.REDIS.scan_counts[-1] == (
        "graphrag:checkpoint:{tenant-1:kb-1:graphrag_checkpoint_community}:keys",
        1,
    )


@pytest.mark.p2
@pytest.mark.asyncio
async def test_save_checkpoint_degrades_on_redis_failure(fake_redis):
    fake_redis.fail_atomic_save = True

    saved = await checkpoints.save_checkpoint("tenant-1", "kb-1", checkpoints.RESOLUTION_CHECKPOINT, "key-1", {"ok": True})

    assert saved is False
    assert fake_redis.values == {}
    assert fake_redis.sets == {}


@pytest.mark.p2
@pytest.mark.asyncio
async def test_cleanup_checkpoints_deletes_redis_stage_keys(fake_redis):
    await checkpoints.save_checkpoint("tenant-1", "kb-1", checkpoints.RESOLUTION_CHECKPOINT, "k1", {"value": 1})
    await checkpoints.save_checkpoint("tenant-1", "kb-1", checkpoints.RESOLUTION_CHECKPOINT, "k2", {"value": 2})
    await checkpoints.save_checkpoint("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT, "k3", {"value": 3})

    cleaned = await checkpoints.cleanup_checkpoints("tenant-1", "kb-1", checkpoints.RESOLUTION_CHECKPOINT, page_size=1)

    assert cleaned is True
    assert await checkpoints.load_checkpoints("tenant-1", "kb-1", checkpoints.RESOLUTION_CHECKPOINT) == {}
    assert await checkpoints.load_checkpoints("tenant-1", "kb-1", checkpoints.COMMUNITY_CHECKPOINT) == {"k3": {"value": 3}}
    assert (
        "graphrag:checkpoint:{tenant-1:kb-1:graphrag_checkpoint_resolution}:keys",
        1,
    ) in fake_redis.REDIS.scan_counts
