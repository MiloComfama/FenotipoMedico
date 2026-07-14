"""Motor y sesión de base de datos (SQLite para el prototipo)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import DB_PATH
from app.db.models import Base

_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Crea las tablas si no existen."""
    Base.metadata.create_all(_engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Sesión transaccional: hace commit al salir, rollback ante error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
