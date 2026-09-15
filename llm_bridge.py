"""
llm_bridge.py — LangChain AI Agent Bridge to Desktop Automation Orchestration Layer

Bridges LLM reasoning models (e.g. ChatOpenAI, ChatAnthropic, Ollama) to the secure desktop
automation orchestration layer.

FEATURES:
1. Tool Schema Integration: Loads tool schemas from configs/execute_ui_action_schema.json
   and configs/read_ui_text_schema.json.
2. System Directive Injection: Loads AI Intent Rationale directive from configs/miku_safety_system_prompt.md.
3. Clean Error Feedback: Catches ActionDeniedError, ElementNotFoundError, and PipelineHaltedError
   and returns formatted feedback to the LLM so it can adjust its reasoning.
4. Fully Async Agent Loop: Invokes AgentLoop.run_task() non-blockingly.
5. Interactive AgentExecutor Loop: Runs live interactive terminal session with ChatOpenAI & AgentExecutor.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from agent_loop import AgentLoop, PipelineHaltedError, TaskSpec
from config import OrchestratorConfig, default_config
from human_gate import ActionDeniedError, ConfirmationTimeoutError, FastConfirm
from safe_executor import ActionDispatch, UnauthorizedActionError
from ui_inspector import ElementNotFoundError, TreeInspector
from visual_highlighter import TargetHighlighter

# Pydantic & LangChain imports with graceful fallback
try:
    from pydantic import BaseModel, Field
    HAVE_PYDANTIC = True
except ImportError:
    HAVE_PYDANTIC = False

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.tools import StructuredTool
    HAVE_LANGCHAIN_CORE = True
except ImportError:
    HAVE_LANGCHAIN_CORE = False

try:
    from langchain.agents import create_agent
    HAVE_LANGCHAIN_AGENTS = True
    AgentExecutor = None
    create_tool_calling_agent = None
except ImportError:
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        HAVE_LANGCHAIN_AGENTS = True
    except ImportError:
        HAVE_LANGCHAIN_AGENTS = False
        AgentExecutor = None
        create_tool_calling_agent = None

try:
    from langchain_openai import ChatOpenAI
    HAVE_OPENAI = True
except ImportError:
    HAVE_OPENAI = False

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
EXECUTE_SCHEMA_PATH = os.path.join(CONFIG_DIR, "execute_ui_action_schema.json")
READ_SCHEMA_PATH = os.path.join(CONFIG_DIR, "read_ui_text_schema.json")
SYSTEM_PROMPT_PATH = os.path.join(CONFIG_DIR, "miku_safety_system_prompt.md")


def load_json_schema(file_path: str) -> Dict[str, Any]:
    """Loads a JSON tool schema definition file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_system_prompt_directive(file_path: str) -> str:
    """Loads the safety system prompt directive."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Define Pydantic Input Schemas for LangChain
if HAVE_PYDANTIC:
    class ExecuteUIActionInput(BaseModel):
        action: str = Field(..., description="The type of physical hardware action: 'click' or 'type'.")
        target: str = Field(..., description="The logical name or AutomationId of the target UI control node.")
        payload: Optional[str] = Field(None, description="Optional text payload to type if action is 'type'.")
        rationale: str = Field(..., description="REQUIRED. Concise first-person explanation of intent explaining why this action is necessary.")

    class ReadScreenTextInput(BaseModel):
        target: str = Field(..., description="Logical name or title of the element, window, or container to read.")
        rationale: str = Field(..., description="REQUIRED. Concise explanation of why you are inspecting screen text.")


class MikuOrchestrationBridge:
    """
    Bridge connecting AI reasoning agents to the desktop automation execution layer.
    """

    def __init__(
        self,
        agent_loop: Optional[AgentLoop] = None,
        config: Optional[OrchestratorConfig] = None,
        simulation_mode: bool = False,
    ):
        self.config = config or default_config
        if agent_loop is None:
            inspector = TreeInspector()
            highlighter = TargetHighlighter(enabled=True)
            gate = FastConfirm(config=self.config, highlighter=highlighter)
            executor = ActionDispatch(simulation_mode=simulation_mode)
            self.agent_loop = AgentLoop(
                inspector=inspector,
                gate=gate,
                executor=executor,
                config=self.config,
            )
        else:
            self.agent_loop = agent_loop

        self.system_directive = load_system_prompt_directive(SYSTEM_PROMPT_PATH)
        self.execute_schema = load_json_schema(EXECUTE_SCHEMA_PATH)
        self.read_schema = load_json_schema(READ_SCHEMA_PATH)

    async def execute_ui_action(
        self,
        action: str,
        target: str,
        rationale: str,
        payload: Optional[str] = None,
    ) -> str:
        """
        Executes a hardware action (click, type) after FastConfirm human authorization.
        Catches safety errors and returns formatted state back to the agent.
        """
        task_spec = TaskSpec(
            action_type=action,
            target_name=target,
            payload=payload,
            rationale=rationale,
        )
        try:
            result = await self.agent_loop.run_task(task_spec)
            if result.get("status") == "success":
                return f"[EXECUTION SUCCESS] Action '{action}' on '{target}' completed successfully."
            else:
                return f"[EXECUTION SKIPPED] Action on '{target}' skipped: {result.get('reason', 'User skipped step')}"
        except ActionDeniedError as e:
            return (
                f"[REJECTED BY HUMAN OPERATOR] The user DENIED your request to '{action}' on '{target}'. "
                f"Reason: {e}. Do not retry this exact action without asking the user for guidance."
            )
        except ConfirmationTimeoutError as e:
            return (
                f"[CONFIRMATION TIMEOUT] Human operator did not respond within timeout: {e}. "
                f"Execution halted."
            )
        except ElementNotFoundError as e:
            return f"[ELEMENT NOT FOUND] Could not locate UI node '{target}' in OS accessibility tree: {e}."
        except PipelineHaltedError as e:
            err_msg = str(e)
            if "denied" in err_msg.lower() or "aborted by user" in err_msg.lower():
                return (
                    f"[REJECTED BY HUMAN OPERATOR] The user DENIED your request to '{action}' on '{target}'. "
                    f"Reason: {e}. Do not retry this action without asking the user for guidance."
                )
            return f"[PIPELINE HALTED] Execution pipeline halted: {e}."
        except Exception as e:
            return f"[ERROR] Execution failed: {e}"

    async def read_screen_text(
        self,
        target: str,
        rationale: str,
    ) -> str:
        """
        Reads screen text from a UI element or window using UIAutomation (bypasses FastConfirm).
        """
        task_spec = TaskSpec(
            action_type="read_screen_text",
            target_name=target,
            rationale=rationale,
        )
        try:
            result = await self.agent_loop.run_task(task_spec)
            if result.get("status") == "success":
                extracted_text = result.get("data", {}).get("text", "")
                return f"[READ SUCCESS] Text retrieved from '{target}':\n{extracted_text}"
            else:
                return f"[READ FAILED] Could not read '{target}': {result.get('reason', 'Unknown error')}"
        except ElementNotFoundError as e:
            return f"[ELEMENT NOT FOUND] Target '{target}' not present in UIA tree: {e}."
        except Exception as e:
            return f"[READ ERROR] Failed to read text: {e}"

    def get_langchain_tools(self) -> List[Any]:
        """Returns LangChain StructuredTools bound to this bridge instance."""
        if not HAVE_LANGCHAIN_CORE:
            raise RuntimeError("langchain_core is not installed. Cannot construct LangChain tools.")

        tools_list = [
            StructuredTool.from_function(
                coroutine=self.execute_ui_action,
                name="execute_ui_action",
                description="Executes a hardware click or typing action on a UI element exposed by OS UIAutomation, guarded by real-time human confirmation.",
                args_schema=ExecuteUIActionInput if HAVE_PYDANTIC else None,
            ),
            StructuredTool.from_function(
                coroutine=self.read_screen_text,
                name="read_screen_text",
                description="Reads text content from a specified UI element or window using Windows UIAutomation without performing hardware manipulation or OCR.",
                args_schema=ReadScreenTextInput if HAVE_PYDANTIC else None,
            ),
        ]
        return tools_list


async def run_offline_local_session(bridge: MikuOrchestrationBridge, prompt: str):
    """
    Executes tasks locally and deterministically without external LLM APIs or pretrained models.
    Satisfies strict offline requirement while testing the full desktop automation pipeline.
    """
    prompt_lower = prompt.lower()
    print(f"\n[LOCAL DETERMINISTIC ENGINE] Processing objective: \"{prompt}\"")

    steps = []
    
    # Check for reading/orienting screen intent
    if "read" in prompt_lower or "orient" in prompt_lower:
        steps.append({
            "tool": "read_screen_text",
            "target": "Desktop",
            "rationale": "Reading current screen state via OS accessibility tree to orient desktop layout."
        })
        
    # Check for start button intent
    if "start" in prompt_lower or "find" in prompt_lower:
        steps.append({
            "tool": "execute_ui_action",
            "action": "click",
            "target": "Start",
            "rationale": "Locating and clicking Windows Start button to open search."
        })

    # Check for notepad / search intent
    if "notepad" in prompt_lower or "search" in prompt_lower:
        steps.append({
            "tool": "execute_ui_action",
            "action": "type",
            "target": "Search",
            "payload": "Notepad\n",
            "rationale": "Typing 'Notepad' into Windows search bar and pressing Enter to launch editor."
        })

    # Check for text insertion intent
    if "miku orchestration layer online" in prompt_lower or "type" in prompt_lower:
        steps.append({
            "tool": "execute_ui_action",
            "action": "type",
            "target": "Text Editor",
            "payload": "Miku orchestration layer online.",
            "rationale": "Writing status confirmation text into the active Notepad document."
        })

    if not steps:
        steps.append({
            "tool": "read_screen_text",
            "target": "Desktop",
            "rationale": "Default screen inspection."
        })

    for idx, step in enumerate(steps, 1):
        print(f"\n--- [LOCAL EXECUTION STEP {idx}/{len(steps)}] ---")
        if step["tool"] == "read_screen_text":
            res = await bridge.read_screen_text(
                target=step["target"],
                rationale=step["rationale"]
            )
            print(res)
        elif step["tool"] == "execute_ui_action":
            res = await bridge.execute_ui_action(
                action=step["action"],
                target=step["target"],
                rationale=step["rationale"],
                payload=step.get("payload")
            )
            print(res)


if __name__ == "__main__":
    import asyncio
    import os

    async def interactive_terminal():
        api_key = os.environ.get("OPENAI_API_KEY")
        
        bridge = MikuOrchestrationBridge()

        if api_key and HAVE_OPENAI and HAVE_LANGCHAIN_AGENTS:
            print("\n" + "="*50)
            print("MIKU AI: LIVE DESKTOP CONTROL INITIALIZED (ONLINE API MODE)")
            print("="*50)
            
            tools = bridge.get_langchain_tools()
            
            if 'create_agent' in globals() and create_agent is not None:
                agent_executor = create_agent(
                    model="gpt-4o",
                    tools=tools,
                    system_prompt=bridge.system_directive
                )
            else:
                llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", bridge.system_directive),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])
                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(
                    agent=agent, 
                    tools=tools, 
                    verbose=True, 
                    handle_parsing_errors=True
                )
            online_mode = True
        else:
            print("\n" + "="*50)
            print("MIKU AI: LIVE DESKTOP CONTROL INITIALIZED (OFFLINE LOCAL MODE)")
            print("="*50)
            print("[SYSTEM] No external API key required / running offline deterministic execution layer.")
            print("[SYSTEM] FastConfirm safety gate + UIA Tree Inspector + Win32 Visual Highlighter ACTIVE.\n")
            online_mode = False

        while True:
            try:
                user_input = input("\nUser> ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    print("\n[SYSTEM] Shutting down Miku orchestration...")
                    break
                
                if not user_input.strip():
                    continue

                if online_mode:
                    if hasattr(agent_executor, 'ainvoke'):
                        await agent_executor.ainvoke({"input": user_input})
                    else:
                        await agent_executor.invoke({"input": user_input})
                else:
                    await run_offline_local_session(bridge, user_input)
                
            except (KeyboardInterrupt, EOFError):
                print("\n[SYSTEM] Live session terminated.")
                break

    asyncio.run(interactive_terminal())


