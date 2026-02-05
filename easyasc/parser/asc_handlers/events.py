from .common import DEvent, SEvent


def handle_create_devent(inst, helper, expr_map) -> None:
    val = inst.kwargs.get("val", None)
    if not isinstance(val, DEvent):
        raise TypeError(f"create_devent需要DEvent类型，当前类型: {type(val)}")
    src_pipe = f"PIPE_{val.src_pipe}"
    dst_pipe = f"PIPE_{val.dst_pipe}"
    preset = "true" if getattr(val, "preset", False) else "false"
    helper(f"DEvent<{src_pipe}, {dst_pipe}, {preset}> {val.name};")


def handle_create_sevent(inst, helper, expr_map) -> None:
    val = inst.kwargs.get("val", None)
    if not isinstance(val, SEvent):
        raise TypeError(f"create_sevent需要SEvent类型，当前类型: {type(val)}")
    src_pipe = f"PIPE_{val.src_pipe}"
    dst_pipe = f"PIPE_{val.dst_pipe}"
    preset = "true" if getattr(val, "preset", False) else "false"
    helper(f"SEvent<{src_pipe}, {dst_pipe}, {preset}> {val.name};")


def handle_event_set(inst, helper, expr_map) -> None:
    event = inst.kwargs.get("event", None)
    if not isinstance(event, (SEvent, DEvent)):
        raise TypeError(f"event_set需要事件类型，当前类型: {type(event)}")
    helper(f"{event.name}.set();")


def handle_event_wait(inst, helper, expr_map) -> None:
    event = inst.kwargs.get("event", None)
    if not isinstance(event, (SEvent, DEvent)):
        raise TypeError(f"event_wait需要事件类型，当前类型: {type(event)}")
    helper(f"{event.name}.wait();")


def handle_event_setall(inst, helper, expr_map) -> None:
    event = inst.kwargs.get("event", None)
    if not isinstance(event, (SEvent, DEvent)):
        raise TypeError(f"event_setall需要事件类型，当前类型: {type(event)}")
    helper(f"{event.name}.setall();")


def handle_event_release(inst, helper, expr_map) -> None:
    event = inst.kwargs.get("event", None)
    if not isinstance(event, (SEvent, DEvent)):
        raise TypeError(f"event_release需要事件类型，当前类型: {type(event)}")
    helper(f"{event.name}.release();")
