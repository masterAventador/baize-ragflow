import threading
from concurrent.futures import ThreadPoolExecutor

from rag.utils import tika_parser


def test_first_tika_parse_is_serialized(monkeypatch):
    """Tika 冷启动只允许一个线程进入，避免多个 Java 进程争抢 9998 端口。"""
    first_entered = threading.Event()
    release_first = threading.Event()
    calls = []

    def fake_from_buffer(blob):
        calls.append(blob)
        if blob == b"first":
            first_entered.set()
            assert release_first.wait(timeout=2)
        return {"content": str(blob)}

    monkeypatch.setattr(tika_parser, "_from_buffer", fake_from_buffer)
    tika_parser._reset_tika_startup_state_for_test()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(tika_parser.from_buffer, b"first")
        assert first_entered.wait(timeout=2)
        second = executor.submit(tika_parser.from_buffer, b"second")

        assert calls == [b"first"]
        release_first.set()
        assert first.result(timeout=2)["content"] == "b'first'"
        assert second.result(timeout=2)["content"] == "b'second'"


def test_tika_parse_skips_startup_lock_after_first_success(monkeypatch):
    """冷启动成功后不再持有初始化锁，保留 Tika 服务自身的并发处理能力。"""
    active = 0
    max_active = 0
    both_entered = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()

    def fake_from_buffer(blob):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                both_entered.set()
        assert release.wait(timeout=2)
        with state_lock:
            active -= 1
        return {"content": str(blob)}

    monkeypatch.setattr(tika_parser, "_from_buffer", lambda blob: {"content": str(blob)})
    tika_parser._reset_tika_startup_state_for_test()
    tika_parser.from_buffer(b"warmup")
    monkeypatch.setattr(tika_parser, "_from_buffer", fake_from_buffer)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(tika_parser.from_buffer, b"first")
        second = executor.submit(tika_parser.from_buffer, b"second")
        assert both_entered.wait(timeout=2)
        release.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert max_active == 2
