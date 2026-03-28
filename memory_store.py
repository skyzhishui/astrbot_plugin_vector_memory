"""
向量记忆存储模块
使用 ChromaDB 实现向量存储
支持：公共记忆、用户专属记忆、秘密记忆
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
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
    """
    
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
        
        logger.info(f"ChromaDB 向量记忆存储初始化完成: {self.db_path}")
        logger.info(f"加载身份映射: {len(self.user_identity_map)} 条，主人列表: {self.masters}")
    
    def get_canonical_user_id(self, session_id: str) -> str:
        """获取用户的规范ID（处理多身份关联）"""
        return self.user_identity_map.get(session_id, session_id)
    
    def is_master(self, session_id: str) -> bool:
        """检查是否为主人"""
        canonical_id = self.get_canonical_user_id(session_id)
        return canonical_id in self.masters
    
    async def add_memory(
        self,
        content: str,
        embedding: list[float],
        category: str = "general",
        importance: float = 0.5,
        source: Optional[str] = None,
        visibility: str = "public",
        owner: Optional[str] = None,
        allowed_users: Optional[list[str]] = None
    ) -> int:
        """添加记忆
        
        Args:
            content: 记忆内容
            embedding: 向量
            category: 类别
            importance: 重要性
            source: 来源
            visibility: 可见性 (public/private/secret)
            owner: 所有者（用户ID）
            allowed_users: 允许访问的用户列表（用于秘密记忆）
        """
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
                        "allowed_users": ",".join(canonical_allowed) if canonical_allowed else ""
                    }]
                )
                
                return memory_id
            
            return await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def search_similar(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0
    ) -> list[dict]:
        """搜索相似记忆（根据用户权限过滤）
        
        Args:
            query_embedding: 查询向量
            user_id: 当前用户ID
            top_k: 返回数量
            category: 类别过滤
            min_importance: 最小重要性
        """
        async with self._lock:
            def _search():
                canonical_user = self.get_canonical_user_id(user_id)
                is_master = self.is_master(user_id)
                
                # 获取更多结果，然后手动过滤权限
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 3,  # 获取更多以便过滤
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
        visibility: Optional[str] = None
    ) -> list[dict]:
        """获取所有记忆（根据用户权限过滤）"""
        async with self._lock:
            def _get():
                canonical_user = self.get_canonical_user_id(user_id)
                is_master = self.is_master(user_id)
                
                results = self.collection.get(include=["documents", "metadatas"])
                
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
                    
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][i],
                        "category": metadata.get("category", "general"),
                        "importance": metadata.get("importance", 0.5),
                        "source": metadata.get("source", ""),
                        "created_at": metadata.get("created_at", ""),
                        "access_count": metadata.get("access_count", 0),
                        "visibility": mem_visibility,
                        "owner": owner
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
                        total_importance += metadata.get("importance", 0.5)
                
                avg_importance = total_importance / count if count > 0 else 0
                
                return {
                    "total_memories": count,
                    "by_category": by_category,
                    "by_visibility": by_visibility,
                    "avg_importance": avg_importance,
                    "user_id": canonical_user,
                    "is_master": is_master
                }
            
            return await asyncio.get_event_loop().run_in_executor(None, _stats)
