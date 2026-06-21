from __future__ import annotations

from app.memory_store import MemoryStore, SalesMemory


def test_memory_store_saves_and_loads_sales_memory(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    memory = SalesMemory(
        user_phone="13800138000",
        device_id="device-1",
        conversation_id="conversation-1",
        intent={"metric": "store_qty", "period": "this_week", "top_n": 0},
        rows=[
            {
                "branch_id": "001",
                "store_name": "徐家汇店",
                "revenue": 1200.0,
                "qty": 42,
                "tickets": 8,
            }
        ],
        answer_summary="徐家汇店售出 42 件。",
    )

    store.save_sales_memory(memory)

    loaded = store.get_sales_memory("13800138000", "device-1", "conversation-1")
    assert loaded is not None
    assert loaded.intent == {"metric": "store_qty", "period": "this_week", "top_n": 0}
    assert loaded.rows[0]["store_name"] == "徐家汇店"
    assert loaded.answer_summary == "徐家汇店售出 42 件。"


def test_memory_store_isolates_sales_memory_by_conversation(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.save_sales_memory(
        SalesMemory(
            user_phone="13800138000",
            device_id="device-1",
            conversation_id="conversation-1",
            intent={"metric": "store_revenue"},
            rows=[{"store_name": "会话一"}],
            answer_summary="会话一",
        )
    )
    store.save_sales_memory(
        SalesMemory(
            user_phone="13800138000",
            device_id="device-1",
            conversation_id="conversation-2",
            intent={"metric": "store_revenue"},
            rows=[{"store_name": "会话二"}],
            answer_summary="会话二",
        )
    )

    first = store.get_sales_memory("13800138000", "device-1", "conversation-1")
    second = store.get_sales_memory("13800138000", "device-1", "conversation-2")

    assert first is not None
    assert second is not None
    assert first.rows[0]["store_name"] == "会话一"
    assert second.rows[0]["store_name"] == "会话二"


def test_memory_store_persists_user_preferences_by_user_and_device(tmp_path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    store = MemoryStore(db_path)
    store.set_preference("13800138000", "device-1", "sales_answer_style", "detailed")
    store.set_preference("13800138000", "device-2", "sales_answer_style", "concise")

    reloaded = MemoryStore(db_path)

    assert reloaded.get_preference("13800138000", "device-1", "sales_answer_style") == "detailed"
    assert reloaded.get_preference("13800138000", "device-2", "sales_answer_style") == "concise"
    assert reloaded.get_preference("13900139000", "device-1", "sales_answer_style") is None
