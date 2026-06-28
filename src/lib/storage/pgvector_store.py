from __future__ import annotations

import json
import logging

from pgvector.psycopg import register_vector
from psycopg import connect

from lib.schemas import EmbeddingRecord

logger = logging.getLogger(__name__)


class PgVectorEmbeddingStore:
    """Base vectorial sobre PostgreSQL + pgvector (provista por la catedra)."""

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        embedding_dim: int = 1280,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.conn = connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            autocommit=True,
        )
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        expected_type = f"vector({self.embedding_dim})"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public'
                  AND c.relname = 'embeddings'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
            )
            row = cur.fetchone()
            if row is not None and row[0] != expected_type:
                logger.warning(
                    "embeddings.embedding is %s but %s is required; dropping table (data removed)",
                    row[0],
                    expected_type,
                )
                cur.execute("DROP TABLE embeddings")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id_imagen TEXT PRIMARY KEY,
                    embedding vector({self.embedding_dim}) NOT NULL,
                    path TEXT NOT NULL,
                    breed TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            ) 

    @staticmethod
    def _row_to_record(row: tuple) -> EmbeddingRecord:
        return EmbeddingRecord(
            id_imagen=row[0],
            embedding=list(row[1]),
            path=row[2],
            breed=row[3],
            metadata=row[4] if isinstance(row[4], dict) else json.loads(row[4]),
        )

    def all(self) -> list[EmbeddingRecord]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id_imagen, embedding, path, breed, metadata
                FROM embeddings
                """
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def append(self, record: EmbeddingRecord) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (id_imagen, embedding, path, breed, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id_imagen) DO NOTHING
                """,
                (
                    record.id_imagen,
                    record.embedding,
                    record.path,
                    record.breed,
                    json.dumps(record.metadata, ensure_ascii=True),
                ),
            )
    #Función para insertar múltiples registros en una sola operación, ignorando duplicados por id_imagen
    #Mejora la eficiencia al reducir la cantidad de conexiones individuales a la base de datos
    #Pasamos de tener una conexión por registro a total_registros / batch_size conexiones
    def append_batch(self, records: list[EmbeddingRecord]) -> None:
        """Inserta múltiples registros en una sola operación, ignorando duplicados por id_imagen."""
        datos = [
            (
                r.id_imagen,
                r.embedding,
                r.path,
                r.breed,
                json.dumps(r.metadata, ensure_ascii=True)
            )
            for r in records
        ]
        
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO embeddings (id_imagen, embedding, path, breed, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id_imagen) DO NOTHING
                """,
                datos
            )

    def search(self, query: list[float], k: int = 10) -> list[EmbeddingRecord]:
        """Top-k vecinos por distancia coseno (operador <=> de pgvector)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id_imagen, embedding, path, breed, metadata
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query, k), 
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]
    
    def alt_search(self, query: list[float], k: int = 10) -> list[EmbeddingRecord]:
        """Top-k vecinos por producto punto (operador <#> de pgvector)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id_imagen, embedding, path, breed, metadata
                FROM embeddings
                ORDER BY embedding <#> %s::vector
                LIMIT %s
                """,
                (query, k),
            )
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    #Función para eliminar todos los registros de la tabla de embeddings
    def truncate(self) -> None:
        """
        Elimina todos los registros de la tabla de embeddings de forma permanente.
        """
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE embeddings;")
            
        logger.info("La tabla 'embeddings' ha sido truncada (vaciada) exitosamente.")