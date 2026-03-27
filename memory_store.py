"""
向量记忆存储模块
使用 SQLite + JSON 实现轻量级向量存储
"""

import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
from astrbot.api import logger


class VectorMemoryStore:
    """向量记忆存储"""
    
    def __init__(self, db_path: str, embedding_dim: int = 1024):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self._lock = asyncio.Lock()
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 记忆表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        ''')
        
        # 创建向量相似度搜索的索引（使用内存中的numpy）
        conn.commit()
        conn.close()
        logger.info(f"向量记忆数据库初始化完成: {self.db_path}")
    
    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """序列化向量为bytes"""
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """反序列化bytes为向量"""
        return np.frombuffer(data, dtype=np.float32)
    
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
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                embedding_bytes = self._serialize_embedding(embedding)
                
                cursor.execute('''
                    INSERT INTO memories (content, embedding, category, importance, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (content, embedding_bytes, category, importance, source, now, now))
                
                memory_id = cursor.lastrowid
                conn.commit()
                conn.close()
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
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # 构建查询
                if category:
                    cursor.execute('''
                        SELECT id, content, embedding, category, importance, source, created_at, access_count
                        FROM memories
                        WHERE category = ? AND importance >= ?
                        ORDER BY importance DESC
                    ''', (category, min_importance))
                else:
                    cursor.execute('''
                        SELECT id, content, embedding, category, importance, source, created_at, access_count
                        FROM memories
                        WHERE importance >= ?
                        ORDER BY importance DESC
                    ''', (min_importance,))
                
                rows = cursor.fetchall()
                conn.close()
                
                if not rows:
                    return []
                
                # 计算相似度
                query_vec = np.array(query_embedding, dtype=np.float32)
                query_norm = np.linalg.norm(query_vec)
                
                results = []
                for row in rows:
                    memory_id, content, embedding_bytes, cat, imp, source, created_at, access_count = row
                    memory_vec = self._deserialize_embedding(embedding_bytes)
                    
                    # 余弦相似度
                    memory_norm = np.linalg.norm(memory_vec)
                    if memory_norm > 0 and query_norm > 0:
                        similarity = np.dot(query_vec, memory_vec) / (query_norm * memory_norm)
                    else:
                        similarity = 0.0
                    
                    results.append({
                        "id": memory_id,
                        "content": content,
                        "category": cat,
                        "importance": imp,
                        "source": source,
                        "created_at": created_at,
                        "access_count": access_count,
                        "similarity": float(similarity)
                    })
                
                # 按相似度排序并返回top_k
                results.sort(key=lambda x: x["similarity"], reverse=True)
                return results[:top_k]
            
            return await asyncio.get_event_loop().run_in_executor(None, _search)
    
    async def get_all_memories(self, category: Optional[str] = None) -> list[dict]:
        """获取所有记忆"""
        async with self._lock:
            def _get():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                if category:
                    cursor.execute('''
                        SELECT id, content, category, importance, source, created_at, access_count
                        FROM memories
                        WHERE category = ?
                        ORDER BY created_at DESC
                    ''', (category,))
                else:
                    cursor.execute('''
                        SELECT id, content, category, importance, source, created_at, access_count
                        FROM memories
                        ORDER BY created_at DESC
                    ''')
                
                rows = cursor.fetchall()
                conn.close()
                
                return [{
                    "id": row[0],
                    "content": row[1],
                    "category": row[2],
                    "importance": row[3],
                    "source": row[4],
                    "created_at": row[5],
                    "access_count": row[6]
                } for row in rows]
            
            return await asyncio.get_event_loop().run_in_executor(None, _get)
    
    async def delete_memory(self, memory_id: int) -> bool:
        """删除记忆"""
        async with self._lock:
            def _delete():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                conn.close()
                return deleted
            
            return await asyncio.get_event_loop().run_in_executor(None, _delete)
    
    async def update_importance(self, memory_id: int, importance: float) -> bool:
        """更新记忆重要性"""
        async with self._lock:
            def _update():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                cursor.execute('''
                    UPDATE memories SET importance = ?, updated_at = ?
                    WHERE id = ?
                ''', (importance, now, memory_id))
                updated = cursor.rowcount > 0
                conn.commit()
                conn.close()
                return updated
            
            return await asyncio.get_event_loop().run_in_executor(None, _update)
    
    async def increment_access_count(self, memory_id: int):
        """增加访问计数"""
        async with self._lock:
            def _increment():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE memories SET access_count = access_count + 1
                    WHERE id = ?
                ''', (memory_id,))
                conn.commit()
                conn.close()
            
            await asyncio.get_event_loop().run_in_executor(None, _increment)
    
    async def get_stats(self) -> dict:
        """获取记忆统计"""
        async with self._lock:
            def _stats():
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM memories')
                total = cursor.fetchone()[0]
                
                cursor.execute('SELECT category, COUNT(*) FROM memories GROUP BY category')
                by_category = dict(cursor.fetchall())
                
                cursor.execute('SELECT AVG(importance) FROM memories')
                avg_importance = cursor.fetchone()[0] or 0
                
                conn.close()
                return {
                    "total_memories": total,
                    "by_category": by_category,
                    "avg_importance": avg_importance
                }
            
            return await asyncio.get_event_loop().run_in_executor(None, _stats)
