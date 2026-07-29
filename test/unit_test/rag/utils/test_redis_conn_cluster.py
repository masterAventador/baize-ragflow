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

"""Redis Cluster 连接与共享集群 Key 命名空间单元测试。"""

from unittest.mock import MagicMock

import pytest

# RAGFlow 现有初始化顺序要求先导入 settings，再由 settings 加载 redis_conn。
from common import settings as _settings  # noqa: F401
from rag.utils import redis_conn


@pytest.mark.p1
def test_parse_cluster_nodes_accepts_comma_separated_addresses():
    nodes = redis_conn._parse_cluster_nodes("redis-1.internal:7001, redis-2.internal:7001")

    assert [(node.host, node.port) for node in nodes] == [
        ("redis-1.internal", 7001),
        ("redis-2.internal", 7001),
    ]


@pytest.mark.p1
def test_parse_cluster_nodes_rejects_invalid_address():
    with pytest.raises(ValueError, match="Redis Cluster 节点地址格式错误"):
        redis_conn._parse_cluster_nodes("redis-1.internal")


@pytest.mark.p1
def test_cluster_client_requires_key_prefix(monkeypatch):
    monkeypatch.setattr(redis_conn.redis, "ValkeyCluster", MagicMock())

    with pytest.raises(ValueError, match="key_prefix"):
        redis_conn._build_redis_client(
            {
                "mode": "cluster",
                "nodes": "redis-1.internal:7001",
                "password": "secret",
                "key_prefix": "",
            }
        )


@pytest.mark.p1
def test_cluster_client_rejects_hash_tag_in_global_prefix(monkeypatch):
    monkeypatch.setattr(redis_conn.redis, "ValkeyCluster", MagicMock())

    with pytest.raises(ValueError, match="不能包含"):
        redis_conn._build_redis_client(
            {
                "mode": "cluster",
                "nodes": "redis-1.internal:7001",
                "password": "secret",
                "key_prefix": "hugai:{prod}:",
            }
        )


@pytest.mark.p1
def test_cluster_client_uses_startup_nodes_without_logical_db(monkeypatch):
    client = object()
    cluster_factory = MagicMock(return_value=client)
    monkeypatch.setattr(redis_conn.redis, "ValkeyCluster", cluster_factory)

    result = redis_conn._build_redis_client(
        {
            "mode": "cluster",
            "nodes": "redis-1.internal:7001,redis-2.internal:7001",
            "username": "hugai",
            "password": "secret",
            "key_prefix": "hugai:knowledge-engine:prod:",
            "db": 5,
        }
    )

    assert result is client
    kwargs = cluster_factory.call_args.kwargs
    assert [(node.host, node.port) for node in kwargs["startup_nodes"]] == [
        ("redis-1.internal", 7001),
        ("redis-2.internal", 7001),
    ]
    assert kwargs["username"] == "hugai"
    assert kwargs["password"] == "secret"
    assert kwargs["decode_responses"] is True
    assert "db" not in kwargs


@pytest.mark.p1
def test_standalone_client_keeps_existing_host_port_and_db(monkeypatch):
    client = object()
    standalone_factory = MagicMock(return_value=client)
    monkeypatch.setattr(redis_conn.redis, "StrictRedis", standalone_factory)

    result = redis_conn._build_redis_client(
        {
            "host": "redis:6379",
            "db": 5,
            "username": "",
            "password": "secret",
        }
    )

    assert result is client
    standalone_factory.assert_called_once_with(
        host="redis",
        port=6379,
        db=5,
        decode_responses=True,
        password="secret",
    )


@pytest.mark.p1
def test_redis_db_prefixes_keys_exactly_once(monkeypatch):
    client = MagicMock()
    client.get.return_value = "value"
    monkeypatch.setattr(redis_conn.REDIS_CONN, "REDIS", client)
    monkeypatch.setattr(
        redis_conn.REDIS_CONN,
        "key_prefix",
        "hugai:knowledge-engine:prod:",
        raising=False,
    )

    assert redis_conn.REDIS_CONN.get("TASKEXE") == "value"
    assert redis_conn.REDIS_CONN.get("hugai:knowledge-engine:prod:TASKEXE") == "value"
    assert client.get.call_args_list == [
        (("hugai:knowledge-engine:prod:TASKEXE",),),
        (("hugai:knowledge-engine:prod:TASKEXE",),),
    ]


@pytest.mark.p1
def test_save_indexed_value_prefixes_both_lua_keys(monkeypatch):
    client = MagicMock()
    script = MagicMock(return_value=1)
    monkeypatch.setattr(redis_conn.REDIS_CONN, "REDIS", client)
    monkeypatch.setattr(
        redis_conn.REDIS_CONN,
        "key_prefix",
        "hugai:knowledge-engine:prod:",
        raising=False,
    )
    monkeypatch.setattr(
        redis_conn.REDIS_CONN,
        "lua_save_indexed_value",
        script,
        raising=False,
    )

    saved = redis_conn.REDIS_CONN.save_indexed_value(
        data_key="graphrag:checkpoint:{tenant:kb:type}:key-1",
        value='{"ok": true}',
        index_key="graphrag:checkpoint:{tenant:kb:type}:keys",
        member="key-1",
        ttl=3600,
    )

    assert saved is True
    script.assert_called_once_with(
        keys=[
            "hugai:knowledge-engine:prod:graphrag:checkpoint:{tenant:kb:type}:key-1",
            "hugai:knowledge-engine:prod:graphrag:checkpoint:{tenant:kb:type}:keys",
        ],
        args=['{"ok": true}', "key-1", 3600],
        client=client,
    )
