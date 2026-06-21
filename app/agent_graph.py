from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from app import business_agents
from app.business_router import BusinessRouteDecision, classify_business_route
from app.config import Settings, get_settings
from app.memory_store import MemoryStore, SalesMemory
from app.models import ChatRequest
from app.openai_client import StreamEvent
from app.sales_tools import OUT_OF_SCOPE_REPLY, format_store_memory_details


@dataclass(frozen=True)
class AgentGraphResult:
    text: str
    route: str
    citations: list[dict[str, str]] = field(default_factory=list)
    used_search: bool = False
    intent: dict[str, Any] | None = None


class AgentGraphState(TypedDict, total=False):
    request: ChatRequest
    client: object
    settings: Settings
    question: str
    messages: list[dict[str, str]]
    decision: BusinessRouteDecision
    route: str
    text: str
    citations: list[dict[str, str]]
    used_search: bool
    intent: dict[str, Any] | None
    user_phone: str


SALES_ASSISTANT_ID = "sales"


async def run_agent_graph(
    request: ChatRequest,
    client: object,
    settings: Settings | None = None,
    user_phone: str = "anonymous",
) -> AgentGraphResult:
    active_settings = settings or get_settings()
    initial: AgentGraphState = {
        "request": request,
        "client": client,
        "settings": active_settings,
        "question": _last_user_question(request),
        "messages": _build_messages(request),
        "citations": [],
        "used_search": False,
        "route": "general_chat",
        "text": "",
        "user_phone": user_phone,
    }
    result = await _compiled_graph().ainvoke(initial)
    return AgentGraphResult(
        text=str(result.get("text", "")),
        route=str(result.get("route", "general_chat")),
        citations=list(result.get("citations", [])),
        used_search=bool(result.get("used_search", False)),
        intent=result.get("intent"),
    )


async def stream_agent_graph_events(
    request: ChatRequest,
    client: object,
    settings: Settings | None = None,
    user_phone: str = "anonymous",
) -> Any:
    result = await run_agent_graph(request, client, settings=settings, user_phone=user_phone)
    if result.text:
        yield StreamEvent(type="delta", text=result.text)
    for citation in result.citations:
        yield StreamEvent(type="citation", citation=citation)
    yield StreamEvent(type="done")


def _compiled_graph() -> Any:
    if not hasattr(_compiled_graph, "_graph"):
        _compiled_graph._graph = _build_graph().compile()  # type: ignore[attr-defined]
    return _compiled_graph._graph  # type: ignore[attr-defined]


def _build_graph() -> StateGraph:
    graph = StateGraph(AgentGraphState)
    graph.add_node("router", _router_node)
    graph.add_node("sales", _sales_node)
    graph.add_node("search", _search_node)
    graph.add_node("general", _general_node)
    graph.add_node("refusal", _refusal_node)
    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        _next_node,
        {
            "sales": "sales",
            "search": "search",
            "general": "general",
            "refusal": "refusal",
        },
    )
    graph.add_edge("sales", END)
    graph.add_edge("search", END)
    graph.add_edge("general", END)
    graph.add_edge("refusal", END)
    return graph


async def _router_node(state: AgentGraphState) -> AgentGraphState:
    request = state["request"]
    question = state.get("question", "")
    settings = state["settings"]
    if _is_sales_detail_followup(question) or _sales_answer_style_preference(question) is not None:
        return {
            "decision": BusinessRouteDecision(
                route="supported_sales",
                reason="sales memory request",
            ),
            "route": "supported_sales",
        }
    decision = await classify_business_route(question, settings=settings)

    if request.assistant_id == SALES_ASSISTANT_ID and decision.route != "supported_sales":
        decision = BusinessRouteDecision(route="unsupported_sales", reason="strict sales mode")

    return {
        "decision": decision,
        "route": decision.route,
    }


def _next_node(state: AgentGraphState) -> Literal["sales", "search", "general", "refusal"]:
    route = state["decision"].route
    if route == "supported_sales":
        return "sales"
    if route == "web_search":
        return "search"
    if route == "general_chat":
        return "general"
    return "refusal"


async def _sales_node(state: AgentGraphState) -> AgentGraphState:
    decision = state["decision"]
    request = state["request"]
    settings = state["settings"]
    memory_store = MemoryStore(settings.memory_db_path)
    user_phone = state.get("user_phone", "anonymous")
    conversation_id = _conversation_id(request)
    preference = _sales_answer_style_preference(state.get("question", ""))

    if preference is not None:
        memory_store.set_preference(
            user_phone,
            request.device_id,
            "sales_answer_style",
            preference,
        )
        label = "详细" if preference == "detailed" else "简洁"
        return {
            "text": f"好的，我已记住：以后这个设备上的销售数据默认用{label}方式回答。",
            "used_search": False,
        }

    if _is_sales_detail_followup(state.get("question", "")):
        memory = memory_store.get_sales_memory(user_phone, request.device_id, conversation_id)
        if memory is None:
            return {
                "text": "我还没有可展开的上一次销售查询。你可以先问一个销售数据问题。",
                "used_search": False,
            }
        return {
            "text": format_store_memory_details(memory.intent, memory.rows),
            "used_search": False,
            "intent": memory.intent,
        }

    if decision.sales_intent is None:
        return {"text": OUT_OF_SCOPE_REPLY, "used_search": False}

    answer_style = memory_store.get_preference(
        user_phone,
        request.device_id,
        "sales_answer_style",
    ) or "concise"
    answer = await business_agents.answer_known_sales_intent(
        decision.sales_intent,
        settings=settings,
        answer_style="detailed" if answer_style == "detailed" else "concise",
    )
    intent_dict = answer.to_dict().get("intent") or {}
    if answer.rows:
        memory_store.save_sales_memory(
            SalesMemory(
                user_phone=user_phone,
                device_id=request.device_id,
                conversation_id=conversation_id,
                intent=dict(intent_dict),
                rows=answer.rows,
                answer_summary=answer.text,
            )
        )
    return {
        "text": answer.text,
        "citations": [],
        "used_search": False,
        "intent": intent_dict,
    }


async def _search_node(state: AgentGraphState) -> AgentGraphState:
    answer = await business_agents.answer_web_search(
        state.get("question", ""),
        state["client"],
        settings=state["settings"],
    )
    return {
        "text": answer.text,
        "citations": answer.citations,
        "used_search": True,
    }


async def _general_node(state: AgentGraphState) -> AgentGraphState:
    text, citations = await business_agents.answer_general_chat(
        state["request"].assistant_id,
        state.get("messages", []),
        state["client"],
    )
    return {
        "text": text,
        "citations": citations,
        "used_search": bool(citations),
    }


async def _refusal_node(state: AgentGraphState) -> AgentGraphState:
    return {
        "text": business_agents.answer_conservative_refusal(state["decision"]),
        "citations": [],
        "used_search": False,
    }


def _last_user_question(request: ChatRequest) -> str:
    return next(
        (message.content for message in reversed(request.messages) if message.role == "user"),
        "",
    ).strip()


def _build_messages(request: ChatRequest) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in request.messages]


def _conversation_id(request: ChatRequest) -> str:
    return request.conversation_id or f"{request.assistant_id}:{request.device_id}"


def _is_sales_detail_followup(question: str) -> bool:
    normalized = question.strip()
    if not normalized:
        return False
    detail_keywords = (
        "详情",
        "详细",
        "展开",
        "展开说",
        "所有门店",
        "全部门店",
        "全部列出来",
        "都列出来",
        "完整",
    )
    return any(keyword in normalized for keyword in detail_keywords)


def _sales_answer_style_preference(question: str) -> str | None:
    normalized = question.strip()
    detailed_markers = (
        "以后都详细",
        "以后详细",
        "默认详细",
        "以后展开",
        "以后都展开",
    )
    concise_markers = (
        "以后都简单",
        "以后简单",
        "以后简洁",
        "默认简洁",
        "默认简单",
    )
    if any(marker in normalized for marker in detailed_markers):
        return "detailed"
    if any(marker in normalized for marker in concise_markers):
        return "concise"
    return None
