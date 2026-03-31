"""
向量记忆存储模块
使用 ChromaDB 实现向量存储
支持：公共记忆、用户专属记忆、秘密记忆、去重机制、分层记忆架构
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Set
from astrbot.api import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB 未安装，请运行: pip install chromadb")


class VectorMemoryStore:
    """向量记忆存储 - ChromaDB 实现
    
    记忆类型 (visibility):
    - public: 公共记忆，所有用户可见（如主人的家庭成员、基本设定）
    - private: 用户专属记忆，仅该用户可见
    - secret: 秘密记忆，仅特定用户触发时可见
    
    分层架构 (layer):
    - L0: 核心层，始终加载（如称呼规则、安全偏好、身份映射）
    - L1: 高频层，关键词触发预加载（如家庭成员、重要日期）
    - L2: 普通层，按需向量检索（如开发规范、配置信息）
    - L3: 归档层，明确询问时才加载
    
    去重机制:
    - 添加记忆前会检查是否存在语义相似的记忆
    - 相似度超过阈值（默认0.95）的记忆会被跳过
    """
    
    # 去重相似度阈值（0-1，越高要求越严格）
    DEDUP_SIMILARITY_THRESHOLD = 0.95
    
    # 分层定义
    LAYERS = {
        "L0": {"desc": "核心层", "priority": 0, "load_strategy": "always"},
        "L1": {"desc": "高频层", "priority": 1, "load_strategy": "keyword"},
        "L2": {"desc": "普通层", "priority": 2, "load_strategy": "vector"},
        "L3": {"desc": "归档层", "priority": 3, "load_strategy": "explicit"}
    }
    
    def __init__(
        self, 
        db_path: str, 
        embedding_dim: int = 1024,
        user_identity_map: Dict[str, str] = None,
        masters: List[str] = None
    ):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB 未安装，请运行: pip install chromadb")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._lock = asyncio.Lock()
        
        # 身份映射：多个 session_id 关联到同一个用户
        # 格式: {"session_id": "canonical_user_id"}
        self.user_identity_map = user_identity_map or {}
        
        # 主人列表（拥有最高权限）
        self.masters = masters or []
        
        # 初始化 ChromaDB 客户端（本地持久化）
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}
        )
        
        # L0/L1 缓存（预热后存储 embedding）
        self._layer_cache: Dict[str, List[dict]] = {}
        self._layer_embeddings: Dict[str, List[List[float]]] = {}
        
        logger.info(f"ChromaDB 向量记忆存储初始化完成: {self.db_path}")
        logger.info(f"加载身份映射: {len(self.user_identity_map)} 条，主人列表: {self.masters}")
        logger.info(f"去重相似度阈值: {self.DEDUP_SIMILARITY_THRESHOLD}")
    
    def get_canonical_user_id(self, session_id: str) -> str:
        """获取用户的规范ID（处理多身份关联）"""
        return self.user_identity_map.get(session_id, session_id)
    
    def is_master(self, session_id: str) -> bool:
        """检查是否为主人"""
        canonical_id = self.get_canonical_user_id(session_id)
        return canonical_id in self.masters
    
    async def warmup_cache(self, get_embedding_func):
        """预热缓存 - 加载 L0 和 L1 层记忆的 embedding
        
        Args:
            get_embedding_func: 异步函数，用于获取文本的 embedding
        """
        logger.info("开始预热 L0/L1 层记忆缓存...")
        
        for layer in ["L0", "L1"]:
            memories = await self.get_layer_memories(layer)
            if memories:
                self._layer_cache[layer] = memories
                self._layer_embeddings[layer] = []
                
                for mem in memories:
                    try:
                        embedding = await get_embedding_func(mem["content"], use_cache=True)
                        self._layer_embeddings[layer].append(embedding)
                    except Exception as e:
                        logger.error(f"预热记忆 {mem['id']} 失败: {e}")
                
                logger.info(f"预热 {layer} 层完成: {len(memories)} 条记忆")
        
        logger.info("缓存预热完成")
    
    async def get_layer_memories(
        self,
        layer: str,
        keywords: List[str] = None
    ) -> List[dict]:
        """获取指定层的记忆
        
        Args:
            layer: 层级 (L0/L1/L2/L3)
            keywords: 关键词列表（用于 L1 层过滤）
        """
        async with self._lock:
            def _get():
                # 使用 where 过滤层级
                results = self.collection.get(
                    where={"layer": layer},
                    include=["documents", "metadatas"]
                )
                
                if not results["ids"]:
                    return []
                
                memories = []
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    
                    # L1 层：关键词过滤
                    if layer == "L1" and keywords:
                        mem_keywords = metadata.get("keywords", "").split(",") if metadata.get("keywords") else []
                        # 如果记忆的关键词与查询关键词有交集，则包含
                        if mem_keywords and not any(kw in mem_keywords for kw in keywords):
                            continue
                    
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][i],
                        "category": metadata.get("category", "general"),
                        "importance": metadata.get("importance", 0.5),
                        "source": metadata.get("source", ""),
                        "created_at": metadata.get("created_at", ""),
                        "access_count": metadata.get("access_count", 0),
                        "visibility": metadata.get("visibility", "public"),
                        "owner": metadata.get("owner", ""),
                        "layer": layer,
                        "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else []
                    })
                
                # 按重要性排序
                memories.sort(key=lambda x: x["importance"], reverse=True)
                return memories
            
            return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def search_in_layer(
        self,
        query_embedding: List[float],
        layer: str,
        top_k: int = 5
    ) -> List[dict]:
        """在指定层内进行向量检索
        
        Args:
            query_embedding: 查询向量
            layer: 层级
            top_k: 返回数量
        """
        async with self._lock:
            def _search():
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where={"layer": layer},
                    include=["documents", "metadatas", "distances"]
                )
                
                if not results["ids"] or not results["ids"][0]:
                    return []
                
                memories = []
                for i, memory_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance
                    
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][0][i],
                        "category": results["metadatas"][0][i].get("category", "general"),
                        "importance": results["metadatas"][0][i].get("importance", 0.5),
                        "similarity": float(similarity),
                        "layer": layer
                    })
                
                return memories
            
            return await asyncio.get_event_loop().run_in_executor(None, _search)
    
    async def _check_duplicate(self, embedding: list[float], content: str, owner: str) -> Optional[int]:
        """检查是否存在重复记忆
        
        Args:
            embedding: 向量
            content: 内容（用于日志）
            owner: 所有者
            
        Returns:
            如果存在重复，返回已存在的记忆ID；否则返回None
        """
        def _check():
            # 搜索相似记忆
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=5,  # 只检查前5个最相似的
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return None
            
            canonical_owner = self.get_canonical_user_id(owner) if owner else None
            
            for i, memory_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                
                # 相似度超过阈值
                if similarity >= self.DEDUP_SIMILARITY_THRESHOLD:
                    metadata = results["metadatas"][0][i]
                    existing_owner = metadata.get("owner", "")
                    existing_content = results["documents"][0][i]
                    
                    # 检查是否是同一用户的记忆（或者都是公共记忆）
                    existing_canonical_owner = self.get_canonical_user_id(existing_owner) if existing_owner else None
                    
                    # 如果所有者相同，或者都是公共记忆，则认为是重复
                    if (canonical_owner == existing_canonical_owner) or \
                       (metadata.get("visibility") == "public" and not existing_owner):
                        logger.info(f"检测到重复记忆 (相似度: {similarity:.3f}):")
                        logger.info(f"  已存在: {existing_content[:50]}...")
                        logger.info(f"  新增: {content[:50]}...")
                        return int(memory_id)
            
            return None
        
        return await asyncio.get_event_loop().run_in_executor(None, _check)
    
    async def add_memory(
        self,
        content: str,
        embedding: list[float],
        category: str = "general",
        importance: float = 0.5,
        source: Optional[str] = None,
        visibility: str = "public",
        owner: Optional[str] = None,
        allowed_users: Optional[list[str]] = None,
        layer: str = "L2",
        keywords: Optional[List[str]] = None,
        skip_duplicate_check: bool = False
    ) -> int:
        """添加记忆（带去重检查）
        
        Args:
            content: 记忆内容
            embedding: 向量
            category: 类别
            importance: 重要性
            source: 来源
            visibility: 可见性 (public/private/secret)
            owner: 所有者（用户ID）
            allowed_users: 允许访问的用户列表（用于秘密记忆）
            layer: 分层 (L0/L1/L2/L3)
            keywords: 关键词列表（用于 L1 层触发）
            skip_duplicate_check: 是否跳过去重检查（默认False）
            
        Returns:
            记忆ID，如果重复则返回已存在记忆的ID（负数表示是重复的）
        """
        # 去重检查
        if not skip_duplicate_check:
            duplicate_id = await self._check_duplicate(embedding, content, owner)
            if duplicate_id is not None:
                logger.info(f"跳过重复记忆，使用已存在的记忆 ID: {duplicate_id}")
                return -duplicate_id  # 返回负数表示是重复的
        
        async with self._lock:
            def _add():
                import time
                import random
                memory_id = int(time.time() * 1000000) + random.randint(0, 999)
                
                now = datetime.now().isoformat()
                
                # 处理 owner 的规范ID
                canonical_owner = self.get_canonical_user_id(owner) if owner else None
                
                # 处理 allowed_users 的规范ID
                canonical_allowed = None
                if allowed_users:
                    canonical_allowed = [self.get_canonical_user_id(u) for u in allowed_users]
                
                # 处理关键词
                keywords_str = ",".join(keywords) if keywords else ""
                
                self.collection.add(
                    ids=[str(memory_id)],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{
                        "category": category,
                        "importance": importance,
                        "source": source or "",
                        "created_at": now,
                        "updated_at": now,
                        "access_count": 0,
                        "visibility": visibility,
                        "owner": canonical_owner or "",
                        "allowed_users": ",".join(canonical_allowed) if canonical_allowed else "",
                        "layer": layer,
                        "keywords": keywords_str
                    }]
                )
                
                # 更新层缓存
                if layer in ["L0", "L1"] and layer in self._layer_cache:
                    self._layer_cache[layer].append({
                        "id": memory_id,
                        "content": content,
                        "category": category,
                        "importance": importance,
                        "layer": layer,
                        "keywords": keywords or []
                    })
                    if layer in self._layer_embeddings:
                        self._layer_embeddings[layer].append(embedding)
                
                return memory_id
            
            return await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def search_similar(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0,
        layer: Optional[str] = None
    ) -> list[dict]:
        """搜索相似记忆（根据用户权限过滤）
        
        Args:
            query_embedding: 查询向量
            user_id: 当前用户ID
            top_k: 返回数量
            category: 类别过滤
            min_importance: 最小重要性
            layer: 层级过滤
        """
        async with self._lock:
            def _search():
                canonical_user = self.get_canonical_user_id(user_id)
                is_master = self.is_master(user_id)
                
                # 构建查询条件
                where_filter = None
                if layer:
                    where_filter = {"layer": layer}
                
                # 获取更多结果，然后手动过滤权限
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 3,  # 获取更多以便过滤
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
                
                if not results["ids"] or not results["ids"][0]:
                    return []
                
                # 格式化并过滤结果
                memories = []
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    
                    # 权限检查
                    visibility = metadata.get("visibility", "public")
                    owner = metadata.get("owner", "")
                    allowed_users = metadata.get("allowed_users", "")
                    
                    # 主人可以看到所有记忆
                    if is_master:
                        pass
                    # 公共记忆：所有人可见
                    elif visibility == "public":
                        pass
                    # 私有记忆：只有所有者可见
                    elif visibility == "private":
                        if owner != canonical_user:
                            continue
                    # 秘密记忆：只有允许的用户可见
                    elif visibility == "secret":
                        if allowed_users and canonical_user not in allowed_users.split(","):
                            continue
                    else:
                        continue
                    
                    # 类别和重要性过滤
                    if category and metadata.get("category") != category:
                        continue
                    if metadata.get("importance", 0.5) < min_importance:
                        continue
                    
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance
                    
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][0][i],
                        "category": metadata.get("category", "general"),
                        "importance": metadata.get("importance", 0.5),
                        "source": metadata.get("source", ""),
                        "created_at": metadata.get("created_at", ""),
                        "access_count": metadata.get("access_count", 0),
                        "visibility": visibility,
                        "owner": owner,
                        "layer": metadata.get("layer", "L2"),
                        "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                        "similarity": float(similarity)
                    })
                    
                    if len(memories) >= top_k:
                        break
                
                return memories
            
            return await asyncio.get_event_loop().run_in_executor(None, _search)
    
    async def get_all_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        visibility: Optional[str] = None,
        layer: Optional[str] = None
    ) -> list[dict]:
        """获取所有记忆（根据用户权限过滤）"""
        async with self._lock:
            def _get():
                canonical_user = self.get_canonical_user_id(user_id)
                is_master = self.is_master(user_id)
                
                # 构建过滤条件
                where_filter = None
                if layer:
                    where_filter = {"layer": layer}
                
                results = self.collection.get(
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
                
                if not results["ids"]:
                    return []
                
                memories = []
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    
                    # 权限检查
                    mem_visibility = metadata.get("visibility", "public")
                    owner = metadata.get("owner", "")
                    allowed_users = metadata.get("allowed_users", "")
                    
                    if is_master:
                        pass
                    elif mem_visibility == "public":
                        pass
                    elif mem_visibility == "private":
                        if owner != canonical_user:
                            continue
                    elif mem_visibility == "secret":
                        if allowed_users and canonical_user not in allowed_users.split(","):
                            continue
                    else:
                        continue
                    
                    # 类别过滤
                    if category and metadata.get("category") != category:
                        continue
                    
                    # 可见性过滤
                    if visibility and mem_visibility != visibility:
                        continue
                    
                    # 层级过滤
                    if layer and metadata.get("layer") != layer:
                        continue
                    
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][i],
                        "category": metadata.get("category", "general"),
                        "importance": metadata.get("importance", 0.5),
                        "source": metadata.get("source", ""),
                        "created_at": metadata.get("created_at", ""),
                        "access_count": metadata.get("access_count", 0),
                        "visibility": mem_visibility,
                        "owner": owner,
                        "layer": metadata.get("layer", "L2"),
                        "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else []
                    })
                
                memories.sort(key=lambda x: x["created_at"], reverse=True)
                return memories
            
            return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def delete_memory(self, memory_id: int, user_id: str) -> bool:
        """删除记忆（需要权限）"""
        async with self._lock:
            def _delete():
                try:
                    # 先检查权限
                    result = self.collection.get(
                        ids=[str(memory_id)],
                        include=["metadatas"]
                    )
                    
                    if not result["ids"]:
                        return False
                    
                    metadata = result["metadatas"][0]
                    owner = metadata.get("owner", "")
                    visibility = metadata.get("visibility", "public")
                    layer = metadata.get("layer", "L2")
                    canonical_user = self.get_canonical_user_id(user_id)
                    
                    # 主人可以删除任何记忆
                    if self.is_master(user_id):
                        pass
                    # 公共记忆只有主人可以删除
                    elif visibility == "public":
                        return False
                    # 私有/秘密记忆只有所有者可以删除
                    elif owner != canonical_user:
                        return False
                    
                    self.collection.delete(ids=[str(memory_id)])
                    
                    # 更新层缓存
                    if layer in self._layer_cache:
                        self._layer_cache[layer] = [
                            m for m in self._layer_cache[layer] if m["id"] != memory_id
                        ]
                    
                    return True
                except Exception as e:
                    logger.error(f"删除记忆失败: {e}")
                    return False
            
            return await asyncio.get_event_loop().run_in_executor(None, _delete)
    
    async def update_importance(self, memory_id: int, importance: float) -> bool:
        """更新记忆重要性"""
        async with self._lock:
            def _update():
                try:
                    result = self.collection.get(
                        ids=[str(memory_id)],
                        include=["documents", "embeddings", "metadatas"]
                    )
                    
                    if not result["ids"]:
                        return False
                    
                    metadata = result["metadatas"][0]
                    metadata["importance"] = importance
                    metadata["updated_at"] = datetime.now().isoformat()
                    
                    self.collection.upsert(
                        ids=[str(memory_id)],
                        embeddings=[result["embeddings"][0]],
                        documents=[result["documents"][0]],
                        metadatas=[metadata]
                    )
                    return True
                except Exception as e:
                    logger.error(f"更新记忆重要性失败: {e}")
                    return False
            
            return await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def update_layer(self, memory_id: int, layer: str) -> bool:
        """更新记忆层级"""
        async with self._lock:
            def _update():
                try:
                    result = self.collection.get(
                        ids=[str(memory_id)],
                        include=["documents", "embeddings", "metadatas"]
                    )
                    
                    if not result["ids"]:
                        return False
                    
                    metadata = result["metadatas"][0]
                    old_layer = metadata.get("layer", "L2")
                    metadata["layer"] = layer
                    metadata["updated_at"] = datetime.now().isoformat()
                    
                    self.collection.upsert(
                        ids=[str(memory_id)],
                        embeddings=[result["embeddings"][0]],
                        documents=[result["documents"][0]],
                        metadatas=[metadata]
                    )
                    
                    # 更新层缓存
                    if old_layer in self._layer_cache:
                        self._layer_cache[old_layer] = [
                            m for m in self._layer_cache[old_layer] if m["id"] != memory_id
                        ]
                    
                    if layer in self._layer_cache:
                        self._layer_cache[layer].append({
                            "id": memory_id,
                            "content": result["documents"][0],
                            "category": metadata.get("category", "general"),
                            "importance": metadata.get("importance", 0.5),
                            "layer": layer,
                            "keywords": metadata.get("keywords", "").split(",") if metadata.get("keywords") else []
                        })
                    
                    return True
                except Exception as e:
                    logger.error(f"更新记忆层级失败: {e}")
                    return False
            
            return await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def increment_access_count(self, memory_id: int):
        """增加访问计数"""
        async with self._lock:
            def _increment():
                try:
                    result = self.collection.get(
                        ids=[str(memory_id)],
                        include=["documents", "embeddings", "metadatas"]
                    )
                    
                    if not result["ids"]:
                        return
                    
                    metadata = result["metadatas"][0]
                    metadata["access_count"] = metadata.get("access_count", 0) + 1
                    
                    self.collection.upsert(
                        ids=[str(memory_id)],
                        embeddings=[result["embeddings"][0]],
                        documents=[result["documents"][0]],
                        metadatas=[metadata]
                    )
                except Exception as e:
                    logger.error(f"更新访问计数失败: {e}")
            
            await asyncio.get_event_loop().run_in_executor(None, _increment)
    
    async def get_stats(self, user_id: str) -> dict:
        """获取记忆统计（根据用户权限）"""
        async with self._lock:
            def _stats():
                canonical_user = self.get_canonical_user_id(user_id)
                is_master = self.is_master(user_id)
                
                results = self.collection.get(include=["metadatas"])
                
                by_category = {}
                by_visibility = {"public": 0, "private": 0, "secret": 0}
                by_layer = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
                total_importance = 0.0
                count = 0
                
                if results["metadatas"]:
                    for metadata in results["metadatas"]:
                        # 权限检查
                        visibility = metadata.get("visibility", "public")
                        owner = metadata.get("owner", "")
                        allowed_users = metadata.get("allowed_users", "")
                        
                        if is_master:
                            pass
                        elif visibility == "public":
                            pass
                        elif visibility == "private":
                            if owner != canonical_user:
                                continue
                        elif visibility == "secret":
                            if allowed_users and canonical_user not in allowed_users.split(","):
                                continue
                        else:
                            continue
                        
                        count += 1
                        cat = metadata.get("category", "general")
                        by_category[cat] = by_category.get(cat, 0) + 1
                        by_visibility[visibility] = by_visibility.get(visibility, 0) + 1
                        
                        layer = metadata.get("layer", "L2")
                        by_layer[layer] = by_layer.get(layer, 0) + 1
                        
                        total_importance += metadata.get("importance", 0.5)
                
                avg_importance = total_importance / count if count > 0 else 0
                
                return {
                    "total_memories": count,
                    "by_category": by_category,
                    "by_visibility": by_visibility,
                    "by_layer": by_layer,
                    "avg_importance": avg_importance,
                    "user_id": canonical_user,
                    "is_master": is_master
                }
            
            return await asyncio.get_event_loop().run_in_executor(None, _stats)
