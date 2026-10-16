from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from dental_agent.workflow.graph import dental_graph

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Dental Appointment Management System                ║
║          Multi-Agent • LangGraph •  (AI)              ║
╚══════════════════════════════════════════════════════════════╝

Examples:
  • Show available slots for an orthodontist
  • Book patient 1000082 with Emily Johnson on 5/10/2026 9:00
  • What appointments does patient 1000048 have?
  • Cancel appointment for patient 1000082 at 5/10/2026 9:00
  • Reschedule patient 1000082 from 5/10/2026 9:00 to 5/12/2026 10:00

Type 'quit' or 'exit' to leave.
"""

def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def run():
    print(BANNER)

    # This list is the conversation history we keep between turns.
    # We only store HumanMessage + AIMessage (final answers).
    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Goodbye!")
            break

        # Add the new user message to history
        history.append(HumanMessage(content=user_input))

        print("\nAgent: ", end="", flush=True)

        final_state = None
        try:
            for data in dental_graph.stream(
                {"messages": history},
                stream_mode="values",
                config={"recursion_limit": 25},
            ):
                final_state = data

        except Exception as exc:
            print(f"\n[Error] {exc}")
            # Roll back the last HumanMessage so the conversation stays clean
            history.pop()
            continue

        if final_state and "messages" in final_state:
            new_messages = final_state["messages"]

            last_ai = None
            for msg in reversed(new_messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    last_ai = msg
                    break

            if last_ai:
                history.append(last_ai)
                answer = message_text(last_ai.content)
                print(answer if answer else "(No text response generated)")
            else:
                print("(No response generated)")
        else:
            print("(No response generated)")


if __name__ == "__main__":
    run()