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

"""Task executor concurrency limiter configuration tests."""

import importlib
import os
from unittest.mock import patch


def test_embedding_concurrency_is_independent_from_chunk_builders():
    from rag.svr import task_executor_limiter

    try:
        with patch.dict(os.environ, {
            "MAX_CONCURRENT_TASKS": "8",
            "MAX_CONCURRENT_CHUNK_BUILDERS": "2",
            "MAX_CONCURRENT_EMBEDDINGS": "8",
        }):
            limiter = importlib.reload(task_executor_limiter)

            assert limiter.task_limiter._value == 8
            assert limiter.chunk_limiter._value == 2
            assert limiter.embed_limiter._value == 8
    finally:
        importlib.reload(task_executor_limiter)
