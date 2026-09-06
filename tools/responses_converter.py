import uuid


def _resp_text_to_blocks(text):
    return [{"type": "input_text", "text": text}]


def responses_to_chat(body):
    """Convert an OpenAI Responses API request to a chat-completions shaped
    equivalent (mirrors llama.cpp's server_chat_convert_responses_to_chatcmpl).

    Returns (chat_body, err). chat_body is a dict understood by parse_request.
    """
    inp = body.get("input")
    if inp is None:
        return None, "`input` is required"
    if body.get("previous_response_id"):
        return None, "`previous_response_id` is not supported"

    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            # responses input item message types (older flat form)
            if role in ("user", "system", "developer", "assistant"):
                content = item.get("content")
                if isinstance(content, str):
                    blocks = _resp_text_to_blocks(content)
                elif isinstance(content, list):
                    blocks = []
                    for blk in content:
                        if not isinstance(blk, dict):
                            continue
                        btype = blk.get("type")
                        if btype == "input_text":
                            blocks.append({"type": "text",
                                           "text": blk.get("text", "")})
                        elif btype == "output_text":
                            blocks.append({"type": "text",
                                           "text": blk.get("text", "")})
                        elif btype == "refusal":
                            blocks.append({"type": "text",
                                           "text": blk.get("refusal", "")})
                        elif btype == "input_image" and blk.get("image_url"):
                            blocks.append({"type": "image_url",
                                           "image_url": {"url": blk["image_url"]}})
                else:
                    blocks = []
                if role == "assistant":
                    # responses output message -> assistant (may carry tool_calls)
                    msg = {"role": "assistant", "content": blocks}
                    calls = item.get("tool_calls")
                    if calls and isinstance(calls, list):
                        tc = []
                        for c in calls:
                            if isinstance(c, dict) and c.get("type") == "function_call":
                                tc.append({
                                    "id": c.get("call_id", f"call_{uuid.uuid4().hex[:12]}"),
                                    "type": "function",
                                    "function": {
                                        "name": c.get("name", ""),
                                        "arguments": c.get("arguments", "{}"),
                                    },
                                })
                        if tc:
                            msg["tool_calls"] = tc
                    messages.append(msg)
                else:
                    messages.append({"role": "user" if role == "developer" else role,
                                     "content": blocks})
                continue
            # typed item form: input_message / output_message / function_call /
            # function_call_output / reasoning
            itype = item.get("type")
            if itype == "input_message":
                role = item.get("role", "user")
                content = item.get("content")
                if isinstance(content, str):
                    blocks = _resp_text_to_blocks(content)
                elif isinstance(content, list):
                    blocks = []
                    for blk in content:
                        if not isinstance(blk, dict):
                            continue
                        btype = blk.get("type")
                        if btype == "input_text":
                            blocks.append({"type": "text",
                                           "text": blk.get("text", "")})
                        elif btype == "output_text":
                            blocks.append({"type": "text",
                                           "text": blk.get("text", "")})
                        elif btype == "refusal":
                            blocks.append({"type": "text",
                                           "text": blk.get("refusal", "")})
                        elif btype == "input_image" and blk.get("image_url"):
                            blocks.append({"type": "image_url",
                                           "image_url": {"url": blk["image_url"]}})
                else:
                    blocks = []
                messages.append({"role": "user" if role in ("developer", "system") else role,
                                 "content": blocks})
            elif itype == "output_message":
                content = item.get("content")
                blocks = []
                if isinstance(content, list):
                    for blk in content:
                        if not isinstance(blk, dict):
                            continue
                        btype = blk.get("type")
                        if btype == "output_text":
                            blocks.append({"type": "text",
                                           "text": blk.get("text", "")})
                        elif btype == "refusal":
                            blocks.append({"type": "text",
                                           "text": blk.get("refusal", "")})
                elif isinstance(content, str):
                    blocks = _resp_text_to_blocks(content)
                msg = {"role": "assistant", "content": blocks}
                # output_message may embed function_call objects
                calls = item.get("tool_calls") or item.get("function_call")
                tc = []
                if isinstance(calls, list):
                    for c in calls:
                        if isinstance(c, dict) and c.get("type") == "function_call":
                            tc.append({"id": c.get("call_id", f"call_{uuid.uuid4().hex[:12]}"),
                                       "type": "function",
                                       "function": {"name": c.get("name", ""),
                                                    "arguments": c.get("arguments", "{}")}})
                elif isinstance(calls, dict):
                    tc.append({"id": calls.get("call_id", f"call_{uuid.uuid4().hex[:12]}"),
                               "type": "function",
                               "function": {"name": calls.get("name", ""),
                                            "arguments": calls.get("arguments", "{}")}})
                if tc:
                    msg["tool_calls"] = tc
                messages.append(msg)
            elif itype == "function_call":
                messages.append({"role": "assistant",
                                 "content": "",
                                 "tool_calls": [{
                                     "id": item.get("call_id", f"call_{uuid.uuid4().hex[:12]}"),
                                     "type": "function",
                                     "function": {"name": item.get("name", ""),
                                                  "arguments": item.get("arguments", "{}")},
                                 }]})
            elif itype == "function_call_output":
                out = item.get("output", "")
                if isinstance(out, list):
                    parts = []
                    for blk in out:
                        if isinstance(blk, dict) and blk.get("type") == "input_text":
                            parts.append(blk.get("text", ""))
                    out = "".join(parts)
                messages.append({"role": "tool",
                                 "tool_call_id": item.get("call_id", ""),
                                 "content": out})
            elif itype == "reasoning":
                content = item.get("content") or []
                if content and isinstance(content, list) and isinstance(content[0], dict):
                    reasoning = content[0].get("text", "")
                    # attach to last assistant message
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1]["reasoning_content"] = reasoning
                    else:
                        messages.append({"role": "assistant", "content": "",
                                         "reasoning_content": reasoning})
    else:
        return None, "`input` must be a string or array"

    # chat body
    chat_body = {
        "messages": messages,
        "stream": bool(body.get("stream", False)),
        "max_tokens": int(body.get("max_output_tokens") or 1024),
        "temperature": float(body.get("temperature", 0.6)),
        "top_p": float(body.get("top_p", 0.95)),
        "top_k": int(body.get("top_k", 20)),
        "seed": body.get("seed"),
        "tool_choice": body.get("tool_choice"),
        "stop": body.get("stop_sequences") or body.get("stop"),
    }
    # tools: responses function tools -> chat tools
    tools = body.get("tools")
    if tools and isinstance(tools, list):
        chat_tools = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") == "function":
                # Responses function tools are flat: name/description/parameters
                # at the top level (llama.cpp maps them via the surrounding dict).
                fn = t.get("function")
                if not isinstance(fn, dict):
                    fn = {"name": t.get("name", ""),
                          "description": t.get("description", ""),
                          "parameters": t.get("parameters",
                                               {"type": "object", "properties": {}})}
                chat_tools.append({"type": "function", "function": fn})
            elif t.get("type") == "custom" and t.get("name"):
                chat_tools.append({"type": "function",
                                   "function": {"name": t.get("name"),
                                                "description": t.get("description", ""),
                                                "parameters": t.get("parameters", {"type": "object",
                                                                                    "properties": {}})}})
        if chat_tools:
            chat_body["tools"] = chat_tools
    # reasoning effort passthrough (ignored by exl3)
    effort = None
    reasoning_cfg = body.get("reasoning")
    if isinstance(reasoning_cfg, dict):
        effort = reasoning_cfg.get("effort")
    if effort is not None:
        chat_body["reasoning_effort"] = effort
    # model
    chat_body["model_id"] = body.get("model", "qwen3.8-27b-exl3-3.5bpw-wm")
    return chat_body, None
