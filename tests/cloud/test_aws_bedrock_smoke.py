import os

import pytest

pytestmark = pytest.mark.cloud
RUN = os.getenv("RUN_AWS_SMOKE") == "true"


@pytest.mark.skipif(not RUN, reason="AWS smoke disabled")
def test_bedrock_chat_smoke() -> None:
    from langchain_aws import ChatBedrock

    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        temperature=0,
    )
    out = llm.invoke("Say 'pong'")
    assert hasattr(out, "content") or isinstance(out, str)
