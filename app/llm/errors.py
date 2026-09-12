class LLMError(Exception):
    """LLM 调用失败的基类。"""


class LLMConnectionError(LLMError):
    """网络不可达或超时。"""


class LLMRateLimitError(LLMError):
    """被限流，通常可重试。"""


class LLMParseError(LLMError):
    """模型返回的内容无法解析成目标结构化模型。"""
