"""
向量记忆存储模块
使用 ChromaDB 实现向量存储
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from astrbot.api import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB 未安装，请运行: pip install chromadb")


class VectorMemoryStore:
    """向量记忆存储 - ChromaDB 实现"""
    
    def __init__(self, db_path: str, embedding_dim: int = 1024):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB 未安装，请运行: pip install chromadb")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._lock = asyncio.Lock()
        
        # 初始化 ChromaDB 客户端（本地持久化）
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,  # 禁用遥测
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        logger.info(f"ChromaDB 向量记忆存储初始化完成: {self.db_path}")
    
    async def add_memory(
        self,
        content: str,
        embedding: list[float],
        category: str = "general",
        importance: float = 0.5,
        source: Optional[str] = None
    ) -> int:
        """添加记忆"""
        async with self._lock:
            def _add():
                # 生成唯一 ID（使用时间戳 + 随机数）
                import time
                import random
                memory_id = int(time.time() * 1000000) + random.randint(0, 999)
                
                now = datetime.now().isoformat()
                
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
                        "access_count": 0
                    }]
                )
                
                return memory_id
            
            return await asyncio.get_event_loop().run_in_executor(None, _add)
    
    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        category: Optional[str] = None,
        min_importance: float = 0.0
    ) -> list[dict]:
        """搜索相似记忆"""
        async with self._lock:
            def _search():
                # 构建过滤条件
                where_filter = None
                if category and min_importance > 0:
                    where_filter = {
                        "$and": [
                            {"category": category},
                            {"importance": {"$gte": min_importance}}
                        ]
                    }
                elif category:
                    where_filter = {"category": category}
                elif min_importance > 0:
                    where_filter = {"importance": {"$gte": min_importance}}
                
                # 执行查询
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
                
                if not results["ids"] or not results["ids"][0]:
                    return []
                
                # 格式化结果
                memories = []
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    # ChromaDB 返回的是距离，转换为相似度 (1 - distance for cosine)
                    distance = results["distances"][0][i]
                    similarity = 1.0 - distance  # 余弦距离转相似度
                    
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][0][i],
                        "category": metadata.get("category", "general"),
                        "importance": metadata.get("importance", 0.5),
                        "source": metadata.get("source", ""),
                        "created_at": metadata.get("created_at", ""),
                        "access_count": metadata.get("access_count", 0),
                        "similarity": float(similarity)
                    })
                
                return memories
            
            return await asyncio.get_event_loop().run_in_executor(None, _search)
    
    async def get_all_memories(self, category: Optional[str] = None) -> list[dict]:
        """获取所有记忆"""
        async with self._lock:
            def _get():
                # 构建过滤条件
                where_filter = {"category": category} if category else None
                
                # 获取所有记录
                results = self.collection.get(
                    where=where_filter,
                    include=["documents", "metadatas"]
                )
                
                if not results["ids"]:
                    return []
                
                # 格式化结果
                memories = []
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    memories.append({
                        "id": int(memory_id),
                        "content": results["documents"][i],
                        "category": metadata.get("category", "general"),
                        "importance": metadata.get("importance", 0.5),
                        "source": metadata.get("source", ""),
                        "created_at": metadata.get("created_at", ""),
                        "access_count": metadata.get("access_count", 0)
                    })
                
                # 按创建时间倒序排列
                memories.sort(key=lambda x: x["created_at"], reverse=True)
                return memories
            
            return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def delete_memory(self, memory_id: int) -> bool:
        """删除记忆"""
        async with self._lock:
            def _delete():
                try:
                    self.collection.delete(ids=[str(memory_id)])
                    return True
                except Exception:
                    return False
            
            return await asyncio.get_event_loop().run_in_executor(None, _delete)
    
    async def update_importance(self, memory_id: int, importance: float) -> bool:
        """更新记忆重要性"""
        async with self._lock:
            def _update():
                try:
                    # ChromaDB 需要先获取再更新
                    result = self.collection.get(
                        ids=[str(memory_id)],
                        include=["documents", "embeddings", "metadatas"]
                    )
                    
                    if not result["ids"]:
                        return False
                    
                    # 更新 metadata
                    metadata = result["metadatas"][0]
                    metadata["importance"] = importance
                    metadata["updated_at"] = datetime.now().isoformat()
                    
                    # 重新 upsert
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
    
    async def get_stats(self) -> dict:
        """获取记忆统计"""
        async with self._lock:
            def _stats():
                # 获取总数
                count = self.collection.count()
                
                # 获取所有记录以统计分类
                results = self.collection.get(include=["metadatas"])
                
                by_category = {}
                total_importance = 0.0
                
                if results["metadatas"]:
                    for metadata in results["metadatas"]:
                        cat = metadata.get("category", "general")
                        by_category[cat] = by_category.get(cat, 0) + 1
                        total_importance += metadata.get("importance", 0.5)
                
                avg_importance = total_importance / count if count > 0 else 0
                
                return {
                    "total_memories": count,
                    "by_category": by_category,
                    "avg_importance": avg_importance
                }
            
            return await asyncio.get_event_loop().run_in_executor(None, _stats)
