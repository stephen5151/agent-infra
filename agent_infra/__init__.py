__all__ = ["build_agent", "run", "stream", "astream", "resume_after_hitl"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        agent_module = importlib.import_module("agent_infra.agent")
        return getattr(agent_module, name)
    raise AttributeError(f"module 'agent_infra' has no attribute {name!r}")
