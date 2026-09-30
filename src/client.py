"""
MCP-Agent-Toolkit  --  app.py  (Streamlit UI)

Same agent as client.py (Groq chat model + MCP client + agent loop),
but with a clean chat interface.

Run:  streamlit run app.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
load_dotenv()

MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 6
SERVER_PATH = str(Path(__file__).parent / "server.py")

SYSTEM_PROMPT = (
    "You are MCP-Agent-Toolkit, a helpful assistant. "
    "Use the available tools to fetch real, current data (weather, GitHub, news). "
    "Never guess live data; call a tool instead. "
    "If you have no tool for something (e.g. forecasts or history), say so honestly. "
    "If a tool returns an error, explain it simply. "
    "Keep answers clear and well formatted."
)

EXAMPLES = [
    "What's the weather in Kota?",
    "Compare the weather in Jaipur, Jodhpur and Udaipur",
    "Show 5 recent repos of GitHub user torvalds",
    "Top 5 Hacker News stories, which is most about AI?",
]

st.set_page_config(page_title="MCP-Agent-Toolkit", page_icon="🧰", layout="centered")

# ---------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .hero {
        padding: 1.2rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        color: white;
      }
      .hero h1 {margin: 0; font-size: 1.7rem; color: white;}
      .hero p {margin: .3rem 0 0 0; opacity: .9; font-size: .95rem;}
      .chip {
        display: inline-block; padding: .15rem .6rem; margin: .15rem .25rem .15rem 0;
        border-radius: 999px; font-size: .78rem; background: rgba(127,127,127,.18);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧰 MCP-Agent-Toolkit</h1>
      <p>An AI agent that calls real APIs through MCP (Model Context Protocol).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# API KEY CHECK
# ---------------------------------------------------------------------
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to your `.env` file and restart.")
    st.stop()

llm = Groq()
server_params = StdioServerParameters(command=sys.executable, args=[SERVER_PATH])


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def to_groq_tools(mcp_tools) -> list[dict]:
    """MCP tool format -> Groq/OpenAI tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def assistant_to_dict(msg) -> dict:
    """Groq message object -> plain dict (so it can live in st.session_state)."""
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments or "{}",
                },
            }
            for c in msg.tool_calls
        ]
    return d


@st.cache_data(show_spinner=False)
def fetch_tool_list() -> list[tuple[str, str]]:
    """Ask the MCP server which tools it has (for the sidebar)."""

    async def _go():
        async with stdio_client(server_params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                listed = await s.list_tools()
                return [(t.name, t.description or "") for t in listed.tools]

    return asyncio.run(_go())


async def run_agent(history: list[dict], status) -> tuple[str, list[dict], list[dict]]:
    """
    Agent loop. Returns (final_answer, new_messages, tool_trace).
    - history: previous messages (without system prompt)
    - status : st.status container to show live progress
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    start = len(messages)
    trace: list[dict] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = to_groq_tools((await session.list_tools()).tools)

            for _ in range(MAX_STEPS):
                resp = llm.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools
                )
                msg = resp.choices[0].message
                messages.append(assistant_to_dict(msg))

                if not msg.tool_calls:
                    return msg.content or "", messages[start:], trace

                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    status.write(f"🔧 Calling `{name}` with `{json.dumps(args)}`")

                    try:
                        out = await session.call_tool(name, args)
                        result = "".join(
                            c.text for c in out.content if hasattr(c, "text")
                        )
                    except Exception as e:
                        result = f"Tool error: {e}"

                    trace.append({"tool": name, "args": args, "result": result})
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": result}
                    )

    return "Step limit reached. Try a simpler question.", messages[start:], trace


def show_trace(trace: list[dict]):
    """Render tool calls inside an expander."""
    if not trace:
        return
    with st.expander(f"🛠️ {len(trace)} tool call(s)"):
        for t in trace:
            st.markdown(f"**`{t['tool']}`**")
            st.code(json.dumps(t["args"], indent=2), language="json")
            try:
                st.json(json.loads(t["result"]))
            except Exception:
                st.code(t["result"][:1500])


# ---------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []      # for display: {"role", "content", "trace"}
if "history" not in st.session_state:
    st.session_state.history = []   # for the LLM: full message dicts

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent")
    st.caption(f"Model: `{MODEL}`")

    st.subheader("Available MCP tools")
    try:
        for name, desc in fetch_tool_list():
            st.markdown(f"<span class='chip'>{name}</span>", unsafe_allow_html=True)
            st.caption(desc)
    except Exception as e:
        st.error(f"Could not load tools: {e}")

    st.subheader("Try these")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------
if not st.session_state.chat:
    st.info("Ask something like: *What's the weather in Kota?* or pick an example from the sidebar.")

for m in st.session_state.chat:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_trace(m.get("trace", []))

# ---------------------------------------------------------------------
# INPUT + RUN
# ---------------------------------------------------------------------
prompt = st.chat_input("Ask about weather, GitHub repos or tech news...")
if not prompt:
    prompt = st.session_state.pop("pending", None)

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        status = st.status("Thinking...", expanded=True)
        try:
            answer, new_msgs, trace = asyncio.run(
                run_agent(st.session_state.history, status)
            )
            status.update(
                label=f"Done ({len(trace)} tool call(s))" if trace else "Done",
                state="complete",
                expanded=False,
            )
            st.markdown(answer)
            show_trace(trace)

            st.session_state.history.extend(new_msgs)
            st.session_state.chat.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )
        except Exception as e:
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.session_state.history.pop()  # failed turn ko history se hatao
            st.session_state.chat.pop()