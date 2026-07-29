#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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

import asyncio
import logging
import json
import uuid

import valkey as redis
from common.decorator import singleton
from common import settings
from valkey.cluster import ClusterNode
from valkey.lock import Lock

REDIS = {}
try:
    REDIS = settings.decrypt_database_config(name="redis")
except Exception:
    try:
        REDIS = settings.get_base_config("redis", {})
    except Exception:
        REDIS = {}

REDIS_MODE_CLUSTER = "cluster"


def _normalize_key_prefix(prefix) -> str:
    """规范化共享 Redis 的 Key 前缀，统一保留末尾冒号。"""
    normalized = str(prefix or "").strip()
    if normalized and not normalized.endswith(":"):
        normalized += ":"
    return normalized


def _prefix_key(prefix: str, key):
    """给 Redis Key 增加一次命名空间前缀，兼容字符串与字节 Key。"""
    if not prefix:
        return key
    if isinstance(key, bytes):
        encoded_prefix = prefix.encode()
        return key if key.startswith(encoded_prefix) else encoded_prefix + key
    normalized_key = str(key)
    return normalized_key if normalized_key.startswith(prefix) else f"{prefix}{normalized_key}"


def _parse_cluster_nodes(raw_nodes) -> list[ClusterNode]:
    """把逗号分隔的 ``host:port`` 启动节点解析为 Valkey ClusterNode。"""
    if isinstance(raw_nodes, str):
        addresses = [item.strip() for item in raw_nodes.split(",") if item.strip()]
    else:
        addresses = list(raw_nodes or [])
    if not addresses:
        raise ValueError("Redis Cluster nodes 不能为空")

    nodes = []
    for address in addresses:
        host, separator, port_text = str(address).strip().rpartition(":")
        try:
            port = int(port_text)
        except (TypeError, ValueError):
            port = 0
        if not separator or not host or not 1 <= port <= 65535:
            raise ValueError(f"Redis Cluster 节点地址格式错误: {address}")
        nodes.append(ClusterNode(host, port))
    return nodes


def _build_redis_client(config: dict):
    """按配置创建单机或 Cluster 客户端，保持单机部署向后兼容。"""
    common_params = {"decode_responses": True}
    username = config.get("username")
    if username:
        common_params["username"] = username
    password = config.get("password")
    if password:
        common_params["password"] = password

    mode = str(config.get("mode", "standalone")).strip().lower()
    if mode == REDIS_MODE_CLUSTER:
        key_prefix = _normalize_key_prefix(config.get("key_prefix"))
        if not key_prefix:
            raise ValueError("共享 Redis Cluster 必须配置 key_prefix")
        if "{" in key_prefix or "}" in key_prefix:
            raise ValueError("Redis Cluster key_prefix 不能包含 { 或 }，否则所有 Key 可能集中到同一 Slot")
        return redis.ValkeyCluster(
            startup_nodes=_parse_cluster_nodes(config.get("nodes")),
            **common_params,
        )

    host, separator, port_text = str(config.get("host", "redis:6379")).rpartition(":")
    if not separator or not host:
        raise ValueError(f"Redis 地址格式错误: {config.get('host')}")
    return redis.StrictRedis(
        host=host,
        port=int(port_text),
        db=int(config.get("db", 1)),
        **common_params,
    )


class RedisMsg:
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        self.__message = json.loads(message["message"])

    def ack(self):
        try:
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]ack" + str(self.__queue_name) + "||" + str(e))
        return False

    def get_message(self):
        return self.__message

    def get_msg_id(self):
        return self.__msg_id


@singleton
class RedisDB:
    lua_delete_if_equal = None
    lua_save_indexed_value = None
    lua_token_bucket = None
    LUA_DELETE_IF_EQUAL_SCRIPT = """
        local current_value = redis.call('get', KEYS[1])
        if current_value and current_value == ARGV[1] then
            redis.call('del', KEYS[1])
            return 1
        end
        return 0
    """

    LUA_TOKEN_BUCKET_SCRIPT = """
        -- KEYS[1] = rate limit key
        -- ARGV[1] = capacity
        -- ARGV[2] = rate
        -- ARGV[3] = now
        -- ARGV[4] = cost

        local key       = KEYS[1]
        local capacity  = tonumber(ARGV[1])
        local rate      = tonumber(ARGV[2])
        local now       = tonumber(ARGV[3])
        local cost      = tonumber(ARGV[4])

        local data = redis.call("HMGET", key, "tokens", "timestamp")
        local tokens = tonumber(data[1])
        local last_ts = tonumber(data[2])

        if tokens == nil then
            tokens = capacity
            last_ts = now
        end

        local delta = math.max(0, now - last_ts)
        tokens = math.min(capacity, tokens + delta * rate)

        if tokens < cost then
            return {0, tokens}
        end

        tokens = tokens - cost

        redis.call("HMSET", key,
            "tokens", tokens,
            "timestamp", now
        )

        redis.call("EXPIRE", key, math.ceil(capacity / rate * 2))

        return {1, tokens}
    """

    LUA_SAVE_INDEXED_VALUE_SCRIPT = """
        -- 两个 Key 必须通过 Redis Cluster Hash Tag 落在同一 Slot。
        redis.call("SET", KEYS[1], ARGV[1], "EX", ARGV[3])
        redis.call("SADD", KEYS[2], ARGV[2])
        redis.call("EXPIRE", KEYS[2], ARGV[3])
        return 1
    """

    def __init__(self):
        self.REDIS = None
        self.config = REDIS
        self.mode = str(self.config.get("mode", "standalone")).strip().lower()
        self.key_prefix = _normalize_key_prefix(self.config.get("key_prefix"))
        self.__open__()

    def register_scripts(self) -> None:
        cls = self.__class__
        client = self.REDIS
        cls.lua_delete_if_equal = client.register_script(cls.LUA_DELETE_IF_EQUAL_SCRIPT)
        cls.lua_save_indexed_value = client.register_script(cls.LUA_SAVE_INDEXED_VALUE_SCRIPT)
        cls.lua_token_bucket = client.register_script(cls.LUA_TOKEN_BUCKET_SCRIPT)

    def __open__(self):
        try:
            self.REDIS = _build_redis_client(self.config)
            self.register_scripts()
        except Exception as e:
            logging.warning(f"Redis can't be connected. Error: {str(e)}")
        return self.REDIS

    def key(self, key):
        """返回带共享集群命名空间的 Redis Key；已带前缀时保持不变。"""
        return _prefix_key(self.key_prefix, key)

    def health(self):
        self.REDIS.ping()
        a, b = "xx", "yy"
        self.set(a, b, 3)

        if self.get(a) == b:
            return True
        return False

    def info(self):
        if self.mode == REDIS_MODE_CLUSTER:
            info = self.REDIS.info(target_nodes=self.REDIS.get_default_node())
        else:
            info = self.REDIS.info()
        return {
            'redis_version': info["redis_version"],
            'server_mode': info["server_mode"] if "server_mode" in info else info.get("redis_mode", ""),
            'used_memory': info["used_memory_human"],
            'total_system_memory': info["total_system_memory_human"],
            'mem_fragmentation_ratio': info["mem_fragmentation_ratio"],
            'connected_clients': info["connected_clients"],
            'blocked_clients': info["blocked_clients"],
            'instantaneous_ops_per_sec': info["instantaneous_ops_per_sec"],
            'total_commands_processed': info["total_commands_processed"]
        }

    def is_alive(self):
        return self.REDIS is not None

    def exist(self, k):
        if not self.REDIS:
            return None
        try:
            return self.REDIS.exists(self.key(k))
        except Exception as e:
            logging.warning("RedisDB.exist " + str(k) + " got exception: " + str(e))
            self.__open__()

    def get(self, k):
        if not self.REDIS:
            return None
        try:
            return self.REDIS.get(self.key(k))
        except Exception as e:
            logging.warning("RedisDB.get " + str(k) + " got exception: " + str(e))
            self.__open__()

    def set_obj(self, k, obj, exp=3600):
        try:
            self.REDIS.set(self.key(k), json.dumps(obj, ensure_ascii=False), exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set_obj " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def set(self, k, v, exp=3600):
        try:
            self.REDIS.set(self.key(k), v, exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def sadd(self, key: str, member: str):
        try:
            self.REDIS.sadd(self.key(key), member)
            return True
        except Exception as e:
            logging.warning("RedisDB.sadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def srem(self, key: str, member: str):
        try:
            self.REDIS.srem(self.key(key), member)
            return True
        except Exception as e:
            logging.warning("RedisDB.srem " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def smembers(self, key: str):
        try:
            res = self.REDIS.smembers(self.key(key))
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.smembers " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def zadd(self, key: str, member: str, score: float):
        try:
            self.REDIS.zadd(self.key(key), {member: score})
            return True
        except Exception as e:
            logging.warning("RedisDB.zadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def zcount(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zcount(self.key(key), min, max)
            return res
        except Exception as e:
            logging.warning("RedisDB.zcount " + str(key) + " got exception: " + str(e))
            self.__open__()
        return 0

    def zpopmin(self, key: str, count: int):
        try:
            res = self.REDIS.zpopmin(self.key(key), count)
            return res
        except Exception as e:
            logging.warning("RedisDB.zpopmin " + str(key) + " got exception: " + str(e))
            self.__open__()
        return None

    def zrangebyscore(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zrangebyscore(self.key(key), min, max)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.zrangebyscore " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def zremrangebyscore(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zremrangebyscore(self.key(key), min, max)
            return res
        except Exception as e:
            logging.warning(
                f"RedisDB.zremrangebyscore {key} got exception: {e}"
            )
            self.__open__()
        return 0

    def zcard(self, key: str):
        try:
            res = self.REDIS.zcard(self.key(key))
            return res
        except Exception as e:
            logging.warning(
                f"RedisDB.zcard {key} got exception: {e}"
            )
            self.__open__()
        return 0

    def incrby(self, key: str, increment: int):
        return self.REDIS.incrby(self.key(key), increment)

    def decrby(self, key: str, decrement: int):
        return self.REDIS.decrby(self.key(key), decrement)

    def zrevrange(self, key: str, start: int, end: int, *, withscores: bool = False):
        """读取倒序 Sorted Set，避免调用方绕过统一 Key 前缀。"""
        return self.REDIS.zrevrange(self.key(key), start, end, withscores=withscores)

    def sscan_iter(self, key: str, *, count: int | None = None):
        """以迭代方式扫描 Set，统一处理共享集群 Key 前缀。"""
        return self.REDIS.sscan_iter(self.key(key), count=count)

    def save_indexed_value(self, *, data_key: str, value: str, index_key: str, member: str, ttl: int) -> bool:
        """原子保存数据及其 Set 索引；调用方需保证两个 Key 使用相同 Hash Tag。"""
        return bool(
            self.lua_save_indexed_value(
                keys=[self.key(data_key), self.key(index_key)],
                args=[value, member, ttl],
                client=self.REDIS,
            )
        )

    def generate_auto_increment_id(self, key_prefix: str = "id_generator", namespace: str = "default",
                                   increment: int = 1, ensure_minimum: int | None = None) -> int:
        redis_key = self.key(f"{key_prefix}:{namespace}")

        try:
            # Use pipeline for atomicity
            pipe = self.REDIS.pipeline()

            # Check if key exists
            pipe.exists(redis_key)

            # Get/Increment
            if ensure_minimum is not None:
                # Ensure minimum value
                pipe.get(redis_key)
                results = pipe.execute()

                if results[0] == 0:  # Key doesn't exist
                    start_id = max(1, ensure_minimum)
                    pipe.set(redis_key, start_id)
                    pipe.execute()
                    return start_id
                else:
                    current = int(results[1])
                    if current < ensure_minimum:
                        pipe.set(redis_key, ensure_minimum)
                        pipe.execute()
                        return ensure_minimum

            # Increment operation
            next_id = self.REDIS.incrby(redis_key, increment)

            # If it's the first time, set a reasonable initial value
            if next_id == increment:
                self.REDIS.set(redis_key, 1 + increment)
                return 1 + increment

            return next_id

        except Exception as e:
            logging.warning("RedisDB.generate_auto_increment_id got exception: " + str(e))
            self.__open__()
        return -1

    def get_or_create_secret_key(self, key_name: str, new_value: str) -> str:
        """
        Atomically get an existing key or create a new one.

        This method guarantees that across multiple concurrent calls, only one
        key will be created and all callers will receive the same key.

        Returns:
            The secret key string

        Raises:
            redis.RedisError: If Redis operations fail
        """
        key_name = self.key(key_name)

        # First, try to get the existing key
        existing_value = self.REDIS.get(key_name)
        if existing_value is not None:
            logging.debug("Retrieved existing key from Redis")
            return existing_value

        # Use SETNX to atomically set the key only if it doesn't exist
        # SETNX returns True if the key was set, False if it already existed
        if self.REDIS.setnx(key_name, new_value):
            logging.info("Successfully created new secret key in Redis")
            return new_value

        # SETNX failed, meaning another process created the key concurrently
        # Retrieve and return that key
        final_key = self.REDIS.get(key_name)
        if final_key is None:
            # This should rarely happen, but retry if it does
            logging.warning("Key disappeared during concurrent access, retrying...")
            return self.get_or_create_secret_key(key_name, new_value)

        logging.debug("Retrieved key created by another process")
        return final_key

    def transaction(self, key, value, exp=3600):
        try:
            return bool(self.REDIS.set(self.key(key), value, exp, nx=True))
        except Exception as e:
            logging.warning(
                "RedisDB.transaction " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return False

    def queue_product(self, queue, message) -> bool:
        queue = self.key(queue)
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                self.REDIS.xadd(queue, payload)
                return True
            except Exception as e:
                logging.exception(
                    "RedisDB.queue_product " + str(queue) + " got exception: " + str(e)
                )
                self.__open__()
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> RedisMsg:
        """https://redis.io/docs/latest/commands/xreadgroup/"""
        queue_name = self.key(queue_name)
        for _ in range(3):
            try:

                try:
                    group_info = self.REDIS.xinfo_groups(queue_name)
                    if not any(gi["name"] == group_name for gi in group_info):
                        self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
                except redis.exceptions.ResponseError as e:
                    if "no such key" in str(e).lower():
                        self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
                    elif "busygroup" in str(e).lower():
                        logging.warning("Group already exists, continue.")
                        pass
                    else:
                        raise

                args = {
                    "groupname": group_name,
                    "consumername": consumer_name,
                    "count": 1,
                    "block": 5,
                    "streams": {queue_name: msg_id},
                }
                messages = self.REDIS.xreadgroup(**args)
                if not messages:
                    return None
                stream, element_list = messages[0]
                if not element_list:
                    return None
                msg_id, payload = element_list[0]
                res = RedisMsg(self.REDIS, queue_name, group_name, msg_id, payload)
                return res
            except Exception as e:
                if str(e) == 'no such key':
                    pass
                else:
                    logging.exception(
                        "RedisDB.queue_consumer "
                        + str(queue_name)
                        + " got exception: "
                        + str(e)
                    )
                    self.__open__()
        return None

    def get_unacked_iterator(self, queue_names: list[str], group_name, consumer_name):
        try:
            for queue_name in queue_names:
                queue_name = self.key(queue_name)
                try:
                    group_info = self.REDIS.xinfo_groups(queue_name)
                except Exception as e:
                    if str(e) == 'no such key':
                        logging.warning(f"RedisDB.get_unacked_iterator queue {queue_name} doesn't exist")
                        continue
                if not any(gi["name"] == group_name for gi in group_info):
                    logging.warning(f"RedisDB.get_unacked_iterator queue {queue_name} group {group_name} doesn't exist")
                    continue
                current_min = 0
                while True:
                    payload = self.queue_consumer(queue_name, group_name, consumer_name, current_min)
                    if not payload:
                        break
                    current_min = payload.get_msg_id()
                    logging.info(f"RedisDB.get_unacked_iterator {queue_name} {consumer_name} {current_min}")
                    yield payload
        except Exception:
            logging.exception(
                "RedisDB.get_unacked_iterator got exception: "
            )
            self.__open__()

    def get_pending_msg(self, queue, group_name):
        queue = self.key(queue)
        try:
            messages = self.REDIS.xpending_range(queue, group_name, '-', '+', 10)
            return messages
        except Exception as e:
            if 'No such key' not in (str(e) or ''):
                logging.warning(
                    "RedisDB.get_pending_msg " + str(queue) + " got exception: " + str(e)
                )
        return []

    def requeue_msg(self, queue: str, group_name: str, msg_id: str):
        queue = self.key(queue)
        for _ in range(3):
            try:
                messages = self.REDIS.xrange(queue, msg_id, msg_id)
                if messages:
                    self.REDIS.xadd(queue, messages[0][1])
                    self.REDIS.xack(queue, group_name, msg_id)
            except Exception as e:
                logging.warning(
                    "RedisDB.get_pending_msg " + str(queue) + " got exception: " + str(e)
                )
                self.__open__()

    def queue_info(self, queue, group_name) -> dict | None:
        queue = self.key(queue)
        for _ in range(3):
            try:
                groups = self.REDIS.xinfo_groups(queue)
                for group in groups:
                    if group["name"] == group_name:
                        return group
            except Exception as e:
                logging.warning(
                    "RedisDB.queue_info " + str(queue) + " got exception: " + str(e)
                )
                self.__open__()
        return None

    def delete_if_equal(self, key: str, expected_value: str) -> bool:
        """
        Do following atomically:
        Delete a key if its value is equals to the given one, do nothing otherwise.
        """
        return bool(self.lua_delete_if_equal(keys=[self.key(key)], args=[expected_value], client=self.REDIS))

    def token_bucket(self, key: str, *, capacity: float, rate: float, now: float, cost: float):
        """执行单 Key Lua 令牌桶，确保共享集群命名空间生效。"""
        return self.lua_token_bucket(
            keys=[self.key(key)],
            args=[capacity, rate, now, cost],
            client=self.REDIS,
        )

    def delete(self, key) -> bool:
        try:
            self.REDIS.delete(self.key(key))
            return True
        except Exception as e:
            logging.warning("RedisDB.delete " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False


REDIS_CONN = RedisDB()


class RedisDistributedLock:
    def __init__(self, lock_key, lock_value=None, timeout=10, blocking_timeout=1):
        self.lock_key = REDIS_CONN.key(lock_key)
        if lock_value:
            self.lock_value = lock_value
        else:
            self.lock_value = str(uuid.uuid4())
        self.timeout = timeout
        self.lock = Lock(REDIS_CONN.REDIS, self.lock_key, timeout=timeout, blocking_timeout=blocking_timeout)

    def acquire(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
        return self.lock.acquire(token=self.lock_value)

    async def spin_acquire(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
        while True:
            if self.lock.acquire(token=self.lock_value):
                break
            await asyncio.sleep(10)

    def release(self):
        REDIS_CONN.delete_if_equal(self.lock_key, self.lock_value)
