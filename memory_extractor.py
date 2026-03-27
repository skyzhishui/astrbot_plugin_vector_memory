"""
记忆提取器
使用LLM从对话中提取重要信息
"""

from astrbot.api import logger


MEMORY_EXTRACTION_PROMPT = """你是一个记忆提取助手。请从以下对话中提取重要的、值得长期记忆的信息。

对话内容:
{conversation}

请提取以下类型的信息（如果存在）:
1. 用户偏好（喜欢/不喜欢的事物）
2. 重要事实（用户提到的重要信息）
3. 个人信息（用户的习惯、特点等）
4. 重要事件（值得记住的事情）
5. 任务和承诺（待办事项、约定等）

请以JSON格式输出，格式如下:
{{
    "memories": [
        {{
            "content": "记忆内容（简洁明了）",
            "category": "preference|fact|personal|event|task",
            "importance": 0.0-1.0
        }}
    ]
}}

注意:
- 只提取真正重要、值得长期记忆的信息
- 避免提取琐碎或临时的信息
- importance越高表示越重要（0-1之间）
- 如果没有值得记忆的信息，返回空列表: {{"memories": []}}

请直接输出JSON，不要有其他内容:"""


class MemoryExtractor:
    """记忆提取器"""
    
    def __init__(self, context):
        self.context = context
    
    async def extract_memories(self, conversation: str, provider_id: str) -> list[dict]:
        """从对话中提取记忆"""
        try:
            prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conversation)
            
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            
            response_text = llm_resp.completion_text.strip()
            
            # 解析JSON响应
            import json
            import re
            
            # 尝试提取JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                memories = result.get("memories", [])
                
                # 过滤掉不重要的记忆
                memories = [m for m in memories if m.get("importance", 0) >= 0.3]
                
                logger.info(f"从对话中提取了 {len(memories)} 条记忆")
                return memories
            
            return []
            
        except Exception as e:
            logger.error(f"提取记忆失败: {e}")
            return []
    
    async def summarize_for_memory(self, user_message: str, assistant_message: str) -> str:
        """将对话总结为记忆格式"""
        return f"用户: {user_message}\n助手: {assistant_message}"
