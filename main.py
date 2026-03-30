"""
AstrBot 向量记忆插件
实现基于向量语义检索的长期记忆系统
"""

import asyncio
import hashlib
from collections import OrderedDict
from pathlib import Path
from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api import AstrBotConfig, logger
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.api.provider import ProviderRequest

from .memory_store import VectorMemoryStore
from .memory_extractor import MemoryExtractor


# 记忆检索提示词
MEMORY_RETRIEVAL_PROMPT = """以下是与你相关的长期记忆，请在回答时参考这些信息:

{memories}

---

请根据这些记忆内容来回答用户的问题，让用户感受到你记得之前说过的事情。"""


class EmbeddingCache:
    """Embedding 缓存（LRU 策略）"""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _hash_text(self, text: str) -> str:
        """计算文本的哈希值作为缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> list[float] | None:
        """从缓存获取 embedding"""
        key = self._hash_text(text)
        if key in self._cache:
            # LRU: 移到末尾表示最近使用
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, text: str, embedding: list[float]):
        """设置缓存"""
        key = self._hash_text(text)
        
        # 如果已存在，更新并移到末尾
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = embedding
            return
        
        # 检查容量，移除最久未使用的
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = embedding
    
    def stats(self) -> dict:
        """获取缓存统计"""
        total = self._hits + self._misses
        hit_rate = self._hits / total * 100 if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


@register(
    name="vector_memory",
    desc="向量化长期记忆系统，支持语义检索的主动记忆",
    version="1.3.1",
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
        self.user_identity_map_list = config.get("user_identity_map", [])
        self.masters = config.get("masters", [])
        
        # Embedding 缓存配置
        self.embedding_cache_size = config.get("embedding_cache_size", 500)
        
        # 解析身份映射
        self.user_identity_map = self._parse_identity_map(self.user_identity_map_list)
        
        # 初始化组件
        self.memory_store: VectorMemoryStore = None
        self.memory_extractor: MemoryExtractor = None
        self.embedding_provider: EmbeddingProvider = None
        
        # Embedding 缓存
        self._embedding_cache: EmbeddingCache = None
        
        # 对话计数器（用于自动记忆）
        self._conversation_counts: dict[str, int] = {}
        
        # 记忆数据库路径
        self.db_path = Path("/AstrBot/data/vector_memory/chroma_db")
        
        logger.info("向量记忆插件初始化中...")
        logger.info(f"解析到身份映射: {self.user_identity_map}")
        logger.info(f"主人列表: {self.masters}")
        logger.info(f"Embedding 缓存大小: {self.embedding_cache_size}")
    
    def _parse_identity_map(self, map_list: list) -> dict:
        """解析身份映射配置列表
        每个元素格式: "userid=sid1,sid2"
        """
        identity_map = {}
        if not map_list:
            return identity_map
        
        for line in map_list:
            line = line.strip()
            if not line or "=" not in line:
                continue
            canonical_id, sids_part = line.split("=", 1)
            canonical_id = canonical_id.strip()
            sids = [s.strip() for s in sids_part.split(",") if s.strip()]
            for sid in sids:
                identity_map[sid] = canonical_id
        
        return identity_map
    
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
                embedding_dim=embedding_dim,
                user_identity_map=self.user_identity_map,
                masters=self.masters
            )
            
            # 初始化记忆提取器
            self.memory_extractor = MemoryExtractor(self.context)
            
            # 初始化 Embedding 缓存
            self._embedding_cache = EmbeddingCache(max_size=self.embedding_cache_size)
            
            logger.info(f"向量记忆插件加载成功！Embedding维度: {embedding_dim}")
            logger.info(f"Embedding 缓存已启用，最大容量: {self.embedding_cache_size}")
            
        except Exception as e:
            logger.error(f"向量记忆插件初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        """获取文本的向量（带缓存）
        
        Args:
            text: 要编码的文本
            use_cache: 是否使用缓存，默认 True
        """
        if not self.embedding_provider:
            raise ValueError("Embedding Provider 未初始化")
        
        # 尝试从缓存获取
        if use_cache and self._embedding_cache:
            cached = self._embedding_cache.get(text)
            if cached is not None:
                logger.debug(f"Embedding 缓存命中: {text[:30]}...")
                return cached
        
        # 调用 API 获取 embedding
        embedding = await self.embedding_provider.get_embedding(text)
        
        # 存入缓存
        if use_cache and self._embedding_cache:
            self._embedding_cache.set(text, embedding)
            logger.debug(f"Embedding 已缓存: {text[:30]}...")
        
        return embedding
    
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """LLM请求前处理 - 自动检索相关记忆并注入"""
        if not self.memory_store:
            return
        
        try:
            # 获取用户消息
            user_message = event.message_str
            user_id = event.get_sender_id()
            if not user_message or len(user_message) < 3:
                return
            
            # 检索相关记忆（使用缓存）
            query_embedding = await self.get_embedding(user_message, use_cache=True)
            memories = await self.memory_store.search_similar(
                query_embedding=query_embedding,
                user_id=user_id,
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
                
                # 将记忆信息注入到系统提示词
                memory_prompt = MEMORY_RETRIEVAL_PROMPT.format(memories=memory_context)
                if request.system_prompt:
                    request.system_prompt += "\n\n" + memory_prompt
                else:
                    request.system_prompt = memory_prompt
            
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
                            embedding = await self.get_embedding(mem["content"], use_cache=True)
                            memory_id = await self.memory_store.add_memory(
                                content=mem["content"],
                                embedding=embedding,
                                category=mem.get("category", "general"),
                                importance=mem.get("importance", 0.5),
                                source=f"session:{session_id}",
                                owner=event.get_sender_id()
                            )
                            # 记录去重情况
                            if memory_id < 0:
                                logger.info(f"自动记忆去重: {mem['content'][:30]}... (已存在ID: {-memory_id})")
                        
                        # 重置计数器
                        self._conversation_counts[session_id] = 0
                        logger.info(f"自动存储了 {len(memories)} 条记忆")
                        
        except Exception as e:
            logger.error(f"自动记忆提取失败: {e}")
    
    # ============ LLM 工具 ============
    
    @filter.llm_tool(name="remember")
    async def tool_remember(self, event: AstrMessageEvent, content: str, category: str = "general", importance: float = 0.5, visibility: str = "private") -> str:
        """主动记住一条信息
        
        Args:
            content(string): 要记住的内容
            category(string): 记忆类别: preference(偏好), fact(事实), personal(个人信息), event(事件), task(任务), general(通用), secret(秘密)
            importance(number): 重要性 0.0-1.0，默认0.5
            visibility(string): 可见性: public(公共)/private(用户专属)/secret(秘密)，默认private
        """
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            embedding = await self.get_embedding(content, use_cache=True)
            memory_id = await self.memory_store.add_memory(
                content=content,
                embedding=embedding,
                category=category,
                importance=importance,
                source=f"user:{event.get_sender_id()}",
                visibility=visibility,
                owner=event.get_sender_id()
            )
            
            # 检查是否是重复记忆（返回负数ID）
            if memory_id < 0:
                return f"⚠️ 检测到重复记忆，已跳过。相似记忆 ID: {-memory_id}"
            
            return f"✓ 已记住: {content} (ID: {memory_id}, 类别: {category}, 可见性: {visibility})"
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
            embedding = await self.get_embedding(query, use_cache=True)
            memories = await self.memory_store.search_similar(
                query_embedding=embedding,
                user_id=event.get_sender_id(),
                top_k=top_k
            )
            
            if not memories:
                return "没有找到相关记忆"
            
            result = f"找到 {len(memories)} 条相关记忆:\n\n"
            for i, m in enumerate(memories, 1):
                result += f"{i}. [{m['category']}] {m['content']}\n"
                result += f"   相关度: {m['similarity']:.2f} | 重要性: {m['importance']:.2f} | 可见性: {m['visibility']}\n"
                # 显示所有者信息
                owner = m.get('owner', '')
                if owner:
                    result += f"   所有者: {owner}\n"
                result += "\n"
            
            return result
        except Exception as e:
            return f"❌ 记忆检索失败: {e}"
    
    @filter.llm_tool(name="list_memories")
    async def tool_list_memories(self, event: AstrMessageEvent, category: str = None, visibility: str = None) -> str:
        """列出所有记忆
        
        Args:
            category(string): 可选，按类别筛选: preference, fact, personal, event, task, general, secret
            visibility(string): 可选，按可见性筛选: public/private/secret
        """
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            memories = await self.memory_store.get_all_memories(
                user_id=event.get_sender_id(),
                category=category,
                visibility=visibility
            )
            
            if not memories:
                return "暂无记忆"
            
            result = f"共有 {len(memories)} 条记忆:\n\n"
            for i, m in enumerate(memories, 1):
                result += f"{i}. [ID:{m['id']}] [{m['category']}] [{m['visibility']}] {m['content']}\n"
                # 显示所有者信息
                owner = m.get('owner', '')
                owner_info = f"所有者: {owner}" if owner else "所有者: 未设置"
                result += f"   重要性: {m['importance']:.2f} | 访问: {m['access_count']}次 | {owner_info}\n\n"
            
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
        
        try:
            deleted = await self.memory_store.delete_memory(
                memory_id=memory_id,
                user_id=event.get_sender_id()
            )
            if deleted:
                return f"✓ 已删除记忆 ID: {memory_id}"
            else:
                return f"❌ 未找到记忆 ID: {memory_id} 或权限不足"
        except Exception as e:
            return f"❌ 删除记忆失败: {e}"
    
    @filter.llm_tool(name="memory_stats")
    async def tool_memory_stats(self, event: AstrMessageEvent) -> str:
        """查看记忆系统统计信息"""
        if not self.memory_store:
            return "❌ 记忆系统未初始化"
        
        try:
            stats = await self.memory_store.get_stats(user_id=event.get_sender_id())
            
            result = "📊 记忆系统统计:\n\n"
            result += f"总记忆数: {stats['total_memories']}\n"
            result += f"平均重要性: {stats['avg_importance']:.2f}\n"
            result += f"当前用户: {stats['user_id']} {'(主人)' if stats['is_master'] else ''}\n\n"
            result += "按类别统计:\n"
            for cat, count in stats['by_category'].items():
                result += f"  • {cat}: {count} 条\n"
            
            result += "\n按可见性统计:\n"
            for vis, count in stats['by_visibility'].items():
                result += f"  • {vis}: {count} 条\n"
            
            # Embedding 缓存统计
            if self._embedding_cache:
                cache_stats = self._embedding_cache.stats()
                result += f"\n📦 Embedding 缓存:\n"
                result += f"  • 缓存大小: {cache_stats['size']}/{cache_stats['max_size']}\n"
                result += f"  • 命中次数: {cache_stats['hits']}\n"
                result += f"  • 未命中次数: {cache_stats['misses']}\n"
                result += f"  • 命中率: {cache_stats['hit_rate']:.1f}%\n"
            
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
            embedding = await self.get_embedding(test_content, use_cache=True)
            memory_id = await self.memory_store.add_memory(
                content=test_content,
                embedding=embedding,
                category="test",
                importance=0.8,
                owner=event.get_sender_id()
            )
            
            # 测试检索
            memories = await self.memory_store.search_similar(embedding, user_id=event.get_sender_id(), top_k=1)
            
            result = f"✓ 记忆系统测试成功!\n"
            result += f"- 存储测试: ID {memory_id}\n"
            result += f"- 检索测试: 找到 {len(memories)} 条记忆\n"
            result += f"- Embedding维度: {len(embedding)}\n"
            result += f"- 用户身份: {self.memory_store.get_canonical_user_id(event.get_sender_id())} {'(主人)' if self.memory_store.is_master(event.get_sender_id()) else ''}"
            
            # 缓存测试
            if self._embedding_cache:
                cache_stats = self._embedding_cache.stats()
                result += f"\n- 缓存命中率: {cache_stats['hit_rate']:.1f}%"
            
            yield event.plain_result(result)
        except Exception as e:
            yield event.plain_result(f"❌ 测试失败: {e}")
    
    @filter.command("memory_cache_clear")
    async def cmd_cache_clear(self, event: AstrMessageEvent):
        """清空 Embedding 缓存"""
        if self._embedding_cache:
            self._embedding_cache.clear()
            yield event.plain_result("✓ Embedding 缓存已清空")
        else:
            yield event.plain_result("❌ 缓存未初始化")
