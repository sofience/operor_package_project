"""
sofi_cli.py — CLI & 데모 진입점
"""

from sofi_agent import agent_step


def main_cli():
    print("=== Sofience_operor-multi-agent-prototype ===")
    print("Ctrl+C 또는 'exit' 입력 시 종료.\n")

    while True:
        try:
            user = input("\n사용자 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("[종료 요청]")
            break

        print("\n[Agent 응답]")
        reply = agent_step(user)
        print(reply)


if __name__ == "__main__":
    main_cli()