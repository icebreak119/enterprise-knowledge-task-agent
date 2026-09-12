from typing import Annotated

from fastapi import Depends

from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider


def get_llm() -> LLMProvider:
    return get_llm_provider()


# 业务代码通过 LLMDep 注入，不感知具体厂商
LLMDep = Annotated[LLMProvider, Depends(get_llm)]
