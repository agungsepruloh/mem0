import contextvars
import json
import os
import re
from typing import Any
from urllib.parse import quote

import anyio
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.responses import Response

load_dotenv()

MEM0_API_URL = os.environ.get("MEM0_API_URL", "http://localhost:8888").rstrip("/")
MEM0_API_KEY = os.environ.get("MEM0_MCP_API_KEY") or os.environ.get("ADMIN_API_KEY")
REQUEST_TIMEOUT_SECONDS = 120.0
SEARCH_TOP_K = 20
LIST_TOP_K = 1000
VALID_MEMORY_SCOPES = {"global", "project"}
PROJECT_ID_PATTERN = re.compile(r"[^a-zA-Z0-9:._/-]+")

mcp = FastMCP("mem0-self-hosted-mcp")
mcp_router = APIRouter(prefix="/mcp")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")
app = FastAPI(title="Mem0 self-hosted MCP adapter")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if MEM0_API_KEY:
        headers["X-API-Key"] = MEM0_API_KEY
    return headers


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_scope(scope: str | None) -> str | None:
    scope = _clean_optional(scope)
    if scope is None:
        return None
    scope = scope.lower()
    if scope not in VALID_MEMORY_SCOPES:
        raise ValueError("scope must be 'global' or 'project'")
    return scope


def _normalize_project_id(project_id: str | None) -> str | None:
    project_id = _clean_optional(project_id)
    if project_id is None:
        return None
    return PROJECT_ID_PATTERN.sub("-", project_id).strip("-") or None


def _build_memory_metadata(
    client_name: str,
    scope: str | None,
    project_id: str | None,
    project_path: str | None,
    project_name: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_app": "openmemory",
        "mcp_client": client_name,
    }
    normalized_scope = _normalize_scope(scope)
    normalized_project_id = _normalize_project_id(project_id)
    if normalized_scope:
        metadata["scope"] = normalized_scope
    if normalized_scope == "project":
        if not normalized_project_id:
            raise ValueError("project_id is required when scope is 'project'")
        metadata["project_id"] = normalized_project_id
    elif normalized_project_id:
        metadata["project_id"] = normalized_project_id
    project_path = _clean_optional(project_path)
    project_name = _clean_optional(project_name)
    if project_path:
        metadata["project_path"] = project_path
    if project_name:
        metadata["project_name"] = project_name
    return metadata


def _matches_scope(metadata: dict[str, Any], scope: str | None, project_id: str | None, include_global: bool) -> bool:
    normalized_scope = _normalize_scope(scope)
    normalized_project_id = _normalize_project_id(project_id)
    memory_scope = metadata.get("scope")
    memory_project_id = metadata.get("project_id")
    if normalized_scope == "global":
        return memory_scope == "global"
    if normalized_scope == "project" or normalized_project_id:
        if memory_scope is None:
            return True
        if include_global and memory_scope == "global":
            return True
        return memory_scope == "project" and memory_project_id == normalized_project_id
    return True


def _results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("results"), list):
        return value["results"]
    if isinstance(value, list):
        return value
    return []


async def _request(method: str, path: str, *, json_body: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.request(method, f"{MEM0_API_URL}{path}", headers=_headers(), json=json_body)
        response.raise_for_status()
        if not response.content:
            return None
        return response.json()


def _context() -> tuple[str | None, str | None]:
    return user_id_var.get(None), client_name_var.get(None)


@mcp.tool(description="Add a new memory. This method is called everytime the user informs anything about themselves, their preferences, or anything that has any relevant information which can be useful in the future conversation. This can also be called when the user asks you to remember something. Set infer to False to store the memory verbatim without LLM fact extraction. Use scope='global' for cross-project user preferences. Use scope='project' with project_id for repo/client/project facts.")
async def add_memories(
    text: str,
    infer: bool = True,
    scope: str | None = None,
    project_id: str | None = None,
    project_path: str | None = None,
    project_name: str | None = None,
) -> str:
    uid, client_name = _context()
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        metadata = _build_memory_metadata(client_name, scope, project_id, project_path, project_name)
        response = await _request(
            "POST",
            "/memories",
            json_body={
                "messages": [{"role": "user", "content": text}],
                "user_id": uid,
                "agent_id": client_name,
                "metadata": metadata,
                "infer": infer,
            },
        )
        return json.dumps(response)
    except Exception as exc:
        return f"Error adding to memory: {exc}"


@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything. Use scope='project' with project_id and include_global=True to recall current project plus global memories.")
async def search_memory(query: str, scope: str | None = None, project_id: str | None = None, include_global: bool = True) -> str:
    uid, client_name = _context()
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        response = await _request(
            "POST",
            "/search",
            json_body={
                "query": query,
                "filters": {"user_id": uid, "agent_id": client_name},
                "top_k": SEARCH_TOP_K,
            },
        )
        results = [
            memory
            for memory in _results(response)
            if _matches_scope(memory.get("metadata") or {}, scope, project_id, include_global)
        ]
        return json.dumps({"results": results}, indent=2)
    except Exception as exc:
        return f"Error searching memory: {exc}"


@mcp.tool(description="List all memories in the user's memory. Optional scope/project_id filters return global memories, project memories, or current project plus global memories.")
async def list_memories(scope: str | None = None, project_id: str | None = None, include_global: bool = True) -> str:
    uid, client_name = _context()
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        response = await _request("GET", f"/memories?user_id={quote(uid)}&agent_id={quote(client_name)}")
        results = [
            memory
            for memory in _results(response)[:LIST_TOP_K]
            if _matches_scope(memory.get("metadata") or {}, scope, project_id, include_global)
        ]
        return json.dumps(results, indent=2)
    except Exception as exc:
        return f"Error getting memories: {exc}"


@mcp.tool(description="Delete specific memories by their IDs")
async def delete_memories(memory_ids: list[str]) -> str:
    deleted = 0
    errors: list[str] = []
    for memory_id in memory_ids:
        try:
            await _request("DELETE", f"/memories/{quote(memory_id)}")
            deleted += 1
        except Exception as exc:
            errors.append(f"{memory_id}: {exc}")
    if errors:
        return f"Deleted {deleted} memories; errors: {'; '.join(errors)}"
    return f"Successfully deleted {deleted} memories"


@mcp.tool(description="Delete all memories in the user's memory")
async def delete_all_memories() -> str:
    uid, client_name = _context()
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        await _request("DELETE", f"/memories?user_id={quote(uid)}&agent_id={quote(client_name)}")
        return "Successfully deleted all memories"
    except Exception as exc:
        return f"Error deleting memories: {exc}"


@mcp.tool(description="Add a memory using the Mem0 hosted-style tool name.")
async def add_memory(text: str, infer: bool = True, app_id: str | None = None) -> str:
    uid, _ = _context()
    client_name = app_id or client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    token = client_name_var.set(client_name)
    try:
        response = await add_memories(text=text, infer=infer)
        return json.dumps({"event_id": "sync", "status": "SUCCEEDED", "result": json.loads(response)})
    except Exception as exc:
        return f"Error adding to memory: {exc}"
    finally:
        client_name_var.reset(token)


@mcp.tool(description="Search memories using the Mem0 hosted-style tool name.")
async def search_memories(query: str, app_id: str | None = None, top_k: int = SEARCH_TOP_K) -> str:
    uid, _ = _context()
    client_name = app_id or client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    try:
        response = await _request(
            "POST",
            "/search",
            json_body={"query": query, "filters": {"user_id": uid, "agent_id": client_name}, "top_k": top_k},
        )
        return json.dumps(response, indent=2)
    except Exception as exc:
        return f"Error searching memory: {exc}"


@mcp.tool(description="List memories using the Mem0 hosted-style tool name.")
async def get_memories(app_id: str | None = None) -> str:
    uid, _ = _context()
    client_name = app_id or client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    response = await _request("GET", f"/memories?user_id={quote(uid)}&agent_id={quote(client_name)}")
    return json.dumps(response, indent=2)


@mcp.tool(description="Get one memory by ID using the Mem0 hosted-style tool name.")
async def get_memory(memory_id: str) -> str:
    response = await _request("GET", f"/memories/{quote(memory_id)}")
    return json.dumps(response, indent=2)


@mcp.tool(description="Update one memory by ID using the Mem0 hosted-style tool name.")
async def update_memory(memory_id: str, text: str, metadata: dict[str, Any] | None = None) -> str:
    response = await _request("PUT", f"/memories/{quote(memory_id)}", json_body={"text": text, "metadata": metadata})
    return json.dumps(response, indent=2)


@mcp.tool(description="Delete one memory by ID using the Mem0 hosted-style tool name.")
async def delete_memory(memory_id: str) -> str:
    return await delete_memories([memory_id])


@mcp.tool(description="Return synchronous event status for hosted-style clients.")
async def get_event_status(event_id: str) -> str:
    return json.dumps({"event_id": event_id, "status": "SUCCEEDED"})


@mcp.tool(description="List entities tracked by the self-hosted Mem0 server.")
async def list_entities() -> str:
    response = await _request("GET", "/entities")
    return json.dumps(response, indent=2)


@mcp.tool(description="Delete an entity tracked by the self-hosted Mem0 server.")
async def delete_entities(entity_type: str, entity_id: str) -> str:
    response = await _request("DELETE", f"/entities/{quote(entity_type)}/{quote(entity_id)}")
    return json.dumps(response, indent=2)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@mcp_router.api_route("/{client_name}/http/{user_id}", methods=["POST", "GET", "DELETE"])
async def handle_streamable_http(request: Request):
    uid = request.path_params.get("user_id")
    client_name = request.path_params.get("client_name")
    user_token = user_id_var.set(uid or "")
    client_token = client_name_var.set(client_name or "")
    response_started = False
    response_status = 200
    response_headers: list[tuple[bytes, bytes]] = []
    response_body = bytearray()

    async def capture_send(message):
        nonlocal response_started, response_status
        if message["type"] == "http.response.start":
            response_started = True
            response_status = message["status"]
            response_headers.extend(message.get("headers", []))
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    try:
        transport = StreamableHTTPServerTransport(
            mcp_session_id=None,
            is_json_response_enabled=True,
        )

        async with anyio.create_task_group() as tg:

            async def run_server(*, task_status=anyio.TASK_STATUS_IGNORED):
                async with transport.connect() as (read_stream, write_stream):
                    task_status.started()
                    await mcp._mcp_server.run(
                        read_stream,
                        write_stream,
                        mcp._mcp_server.create_initialization_options(),
                        stateless=True,
                    )

            await tg.start(run_server)
            await transport.handle_request(request.scope, request.receive, capture_send)
            await transport.terminate()
            tg.cancel_scope.cancel()
    finally:
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)

    if not response_started:
        return Response(status_code=500, content=b"Transport did not produce a response")
    return Response(
        content=bytes(response_body),
        status_code=response_status,
        headers={k.decode(): v.decode() for k, v in response_headers},
    )


app.include_router(mcp_router)
