"""
AstrBot 向量记忆插件
实现基于向量语义检索的长期记忆系统
"""

import asyncio
from pathlib import Path
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api import AstrBotConfig, logger
from astrbot.core.provider.provider import EmbeddingProvider

from .memory_store import VectorMemoryStore
from .memory_extractor import MemoryExtractor


# 记忆检索提示词
MEMORY_RETRIEVAL_PROMPT = """以下是与你相关的长期记忆，请在回答时参考这些信息:

{memories}

---

请根据这些记忆内容来回答用户的问题，让用户感受到你记得之前说过的事情。"""


@register(
    name="vector_memory",
    desc="向量化长期记忆系统，支持语义检索的主动记忆",
    version="1.0.0",
    author="Neko"
)
class VectorMemoryPlugin(Star):
    """向量记忆插件"""
    
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 配置参数
        self.embedding_provider_id = config.get("embedding_provider_id", "")
        self.auto_remember = config.get("auto_remember", True)
        self.memory_threshold = config.get("memory_threshold", 5)
        self.top_k = config.get("top_k", 5)
        self.admin_ids = set(config.get("admin_ids", []))
        
        # 初始化组件
        self.memory_store: VectorMemoryStore = None
        self.memory_extractor: MemoryExtractor = None
        self.embedding_provider: EmbeddingProvider = None
        
        # 对话计数器（用于自动记忆）
        self._conversation_counts: dict[str, int] = {}
        
        # 记忆数据库路径
        self.db_path = Path("/AstrBot/data/vector_memory/memories.db")
        
        logger.info("向量记忆插件初始化中...")
    
    async def initialize(self):
        """插件激活时调用"""
        try:
            # 获取 Embedding Provider
            if not self.embedding_provider_id:
                logger.warning("未配置 embedding_provider_id，请在配置中设置")
                return
            
            # 从 Provider Manager 获取 Embedding Provider
            for provider in self.context.provider_manager.embedding_provider_insts:
                if provider.meta().id == self.embedding_provider_id:
                    self.embedding_provider = provider
                    break
            
            if not self.embedding_provider:
                logger.error(f"未找到 Embedding Provider: {self.embedding_provider_id}")
                return
            
            # 初始化向量存储
            embedding_dim = self.embedding_provider.get_dim()
            self.memory_store = VectorMemoryStore(
                db_path=str(self.db_path),
                embedding_dim=embedding_dim
            )
            
            # 初始化记忆提取器
            self.memory_extractor = MemoryExtractor(self.context)
            
            logger.info(f"向量记忆插件加载成功！Embedding维度: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"向量记忆插件初始化失败: {e}")
    
    def is_admin(self, user_id: str) -> bool:
        """检查是否为管理员"""
        if not self.admin_ids:
            return True  # 如果未配置管理员，所有人都是管理员
        return str(user_id) in self.admin_ids
    
    async def get_embedding(self, text: str) -> list[float]:
        """获取文本的向量"""
        if not self.embedding_provider:
            raise ValueError("Embedding Provider 未初始化")
        return await self.embedding_provider.get_embedding(text)
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent):
        """LLM请求前处理 - 自动检索相关记忆并注入"""
        if not self.memory_store:
            return
        
        try:
            # 获取用户消息
            user_message = event.message_str
            if not user_message or len(user_message) < 3:
                return
            
            # 检索相关记忆
            query_embedding = await self.get_embedding(user_message)
            memories = await self.memory_store.search_similar(
                query_embedding=query_embedding,
                top_k=self.top_k
            )
            
            # 如果有相关记忆，注入到上下文
            if memories:
                memory_texts = []
                for m in memories:
                    memory_texts.append(f"[{m['category']}] {m['content']} (相关度: {m['similarity']:.2f})")
                    # 增加访问计数
                    await self.memory_store.increment_access_count(m['id'])
                
                memory_context = "\n".join(memory_texts)
                logger.info(f"检索到 {len(memories)} 条相关记忆")
                
                # 将记忆信息存储到事件额外数据中，供后续使用
                event.set_extra("vector_memory_context", memory_context)
            
            # 更新对话计数
            session_id = event.unified_msg_origin
            if session_id not in self._conversation_counts:
                self._conversation_counts[session_id] = 0
            self._conversation_counts[session_id] += 1
            
        except Exception as e:
            logger.error(f"记忆检索失败: {e}")
    
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, response):
        """LLM响应后处理 - 自动记忆提取"""
        if not self.auto_remember or not self.memory_store or not self.memory_extractor:
            return
        
        try:
            session_id = event.unified_msg_origin
            
            # 检查是否达到记忆阈值
            if self._conversation_counts.get(session_id, 0) >= self.memory_threshold:
                # 获取当前对话历史
                conv_mgr = self.context.conversation_manager
                curr_cid = await conv_mgr.get_curr_conversation_id(session_id)
                
                if curr_cid:
                    conversation = await conv_mgr.get_conversation(session_id, curr_cid)
                    if conversation and conversation.history:
                        # 提取记忆
                        provider_id = await self.context.get_current_chat_provider_id(session_id)
                        memories = await self.memory_extractor.extract_memories(
                            conversation.history,
                            provider_id
                        )
                        
                        # 存储记忆
                        for mem in memories:
                            embedding = await self.get_embedding(mem["content"])
                            await self.memory_store.add_memory(
                                content=mem["content"],
                                embedding=embedding,
                                category=mem.get("category", "general"),
                                importance=mem.get("importance", 0.5),
                                source=f"session:{session_id}"
                            )
                        
                        # 重置计数器
                        self._conversation_counts[session_id] = 0
                        logger.info(f"自动存储了 {len(memories)} 条记忆")
                        
        except Exception as e:
            logger.error(f"自动记忆提取失败: {e}")
    
    # ============ LLM 工具 ============
    
    @filter.llm_tool(name="remember")
    async def tool_remember(self, event: AstrMessageEvent, content: str, category: str = "general", importance: float = 0.5) -> str:
        """主动记住一条信息
        
        Args:
            content(string): 要记住的内容
            category(string): 记忆类别: preference(偏好), fact(事实), personal(个人信息), event(事件), task(任务), general(通用)
            importance(number): 重要性 0.0-1.0，默认0.5
        """
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            embedding = await self.get_embedding(content)
            memory_id = await self.memory_store.add_memory(
                content=content,
                embedding=embedding,
                category=category,
                importance=importance,
                source=f"user:{event.get_sender_id()}"
            )
            return f"✓ 已记住: {content} (ID: {memory_id}, 类别: {category})"
        except Exception as e:
            return f"❌ 记忆存储失败: {e}"
    
    @filter.llm_tool(name="recall_memories")
    async def tool_recall_memories(self, event: AstrMessageEvent, query: str, top_k: int = 5) -> str:
        """搜索相关记忆
        
        Args:
            query(string): 搜索关键词或问题
            top_k(number): 返回的记忆数量，默认5
        """
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            embedding = await self.get_embedding(query)
            memories = await self.memory_store.search_similar(
                query_embedding=embedding,
                top_k=top_k
            )
            
            if not memories:
                return "没有找到相关记忆"
            
            result = f"找到 {len(memories)} 条相关记忆:\n\n"
            for i, m in enumerate(memories, 1):
                result += f"{i}. [{m['category']}] {m['content']}\n"
                result += f"   相关度: {m['similarity']:.2f} | 重要性: {m['importance']:.2f}\n\n"
            
            return result
        except Exception as e:
            return f"❌ 记忆检索失败: {e}"
    
    @filter.llm_tool(name="list_memories")
    async def tool_list_memories(self, event: AstrMessageEvent, category: str = None) -> str:
        """列出所有记忆
        
        Args:
            category(string): 可选，按类别筛选: preference, fact, personal, event, task, general
        """
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            memories = await self.memory_store.get_all_memories(category)
            
            if not memories:
                return "暂无记忆"
            
            result = f"共有 {len(memories)} 条记忆:\n\n"
            for i, m in enumerate(memories, 1):
                result += f"{i}. [ID:{m['id']}] [{m['category']}] {m['content']}\n"
                result += f"   重要性: {m['importance']:.2f} | 访问: {m['access_count']}次\n\n"
            
            return result
        except Exception as e:
            return f"❌ 获取记忆列表失败: {e}"
    
    @filter.llm_tool(name="forget_memory")
    async def tool_forget_memory(self, event: AstrMessageEvent, memory_id: int) -> str:
        """删除一条记忆
        
        Args:
            memory_id(number): 要删除的记忆ID
        """
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        if not self.is_admin(event.get_sender_id()):
            return "❌ 权限不足，只有管理员可以删除记忆"
        
        try:
            deleted = await self.memory_store.delete_memory(memory_id)
            if deleted:
                return f"✓ 已删除记忆 ID: {memory_id}"
            else:
                return f"❌ 未找到记忆 ID: {memory_id}"
        except Exception as e:
            return f"❌ 删除记忆失败: {e}"
    
    @filter.llm_tool(name="memory_stats")
    async def tool_memory_stats(self, event: AstrMessageEvent) -> str:
        """查看记忆系统统计信息"""
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            stats = await self.memory_store.get_stats()
            
            result = "📊 记忆系统统计:\n\n"
            result += f"总记忆数: {stats['total_memories']}\n"
            result += f"平均重要性: {stats['avg_importance']:.2f}\n\n"
            result += "按类别统计:\n"
            for cat, count in stats['by_category'].items():
                result += f"  • {cat}: {count} 条\n"
            
            return result
        except Exception as e:
            return f"❌ 获取统计信息失败: {e}"
    
    # ============ 命令 ============
    
    @filter.command("memory_init")
    async def cmd_memory_init(self, event: AstrMessageEvent):
        """初始化记忆系统"""
        await self.initialize()
        if self.memory_store:
            yield event.plain_result("✓ 记忆系统初始化成功！")
        else:
            yield event.plain_result("❌ 记忆系统初始化失败，请检查配置")
    
    @filter.command("memory_test")
    async def cmd_memory_test(self, event: AstrMessageEvent):
        """测试记忆系统"""
        if not self.memory_store:
            yield event.plain_result("❌ 记忆系统未初始化，请先运行 /memory_init")
            return
        
        try:
            # 测试存储
            test_content = "这是一条测试记忆"
            embedding = await self.get_embedding(test_content)
            memory_id = await self.memory_store.add_memory(
                content=test_content,
                embedding=embedding,
                category="test",
                importance=0.8
            )
            
            # 测试检索
            memories = await self.memory_store.search_similar(embedding, top_k=1)
            
            result = f"✓ 记忆系统测试成功!\n"
            result += f"- 存储测试: ID {memory_id}\n"
            result += f"- 检索测试: 找到 {len(memories)} 条记忆\n"
            result += f"- Embedding维度: {len(embedding)}"
            
            yield event.plain_result(result)
        except Exception as e:
            yield event.plain_result(f"❌ 测试失败: {e}")
