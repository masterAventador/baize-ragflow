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
"""Helpers for the delete-documents endpoint.

Kept dependency-free (no quart / DB imports) so the ownership-validation logic
can be unit-tested without booting the web app.

**HugAI performance patch background** — the delete endpoint originally checked
"do these ids belong to the dataset?" by materialising EVERY document of the
dataset via the ORM and building a full id set:

    dataset_doc_ids = {doc.id for doc in DocumentService.query(kb_id=dataset_id)}

That is O(dataset size), independent of how many ids you actually delete. On a
760k-document knowledge base a single call takes ~49s (2.4s of SQL + ~20x ORM
object construction) and, because the ragflow API is single-process with this
query running synchronously inside the async handler, it wedges the whole API —
concurrent retrieval/list requests starve for the duration. The fix pushes the
requested ids down into a targeted `WHERE kb_id=? AND id IN (...)` lookup
(O(request size), index-backed, milliseconds) via ``query_existing`` while
keeping the exact same invalid-id semantics and error message.
"""

from typing import Callable, Iterable


def find_invalid_doc_ids(
    dataset_id: str,
    doc_ids: list[str],
    query_existing: Callable[[str, list[str]], Iterable[str]],
) -> list[str]:
    """Return the requested ids that do NOT belong to ``dataset_id``.

    ``query_existing(dataset_id, doc_ids)`` must return the subset of ``doc_ids``
    that actually exist under this dataset — implementations are expected to run a
    targeted query over the requested ids only, never a full-dataset scan. Order
    of the returned invalid list follows the request order so error messages stay
    reproducible (matches the previous whole-table implementation's behaviour).
    """
    existing = set(query_existing(dataset_id, doc_ids))
    return [doc_id for doc_id in doc_ids if doc_id not in existing]
