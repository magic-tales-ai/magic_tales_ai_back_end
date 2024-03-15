import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession


# Use an async-compatible database URL, for example for MySQL:
# mysql+aiomysql://<user>:<password>@<host>/<dbname>
SQLALCHEMY_DATABASE_URL = f"mysql+aiomysql://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"

# Create async engine
async_engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=True)


@asynccontextmanager
async def transaction_context(session: AsyncSession) -> AsyncGenerator[None, None]:
    """Context manager to for committing a transaction.

    At the begging of the scope of the context, a transaction will be started,
    unless one is already in progress (which is most likely the case).

    When the scope of the context ends, a commit will be attempted. If any error
    happens within the scope of the context, the transaction will be rolled back and
    an error will be raised.

    This is, in essence, the same as ``async with session.begin():``, except that it
    will not throw an error if a transaction is already in process.

    See
    https://github.com/sqlalchemy/sqlalchemy/discussions/6864#discussioncomment-1142958
    for more information.
    """
    if not session.in_transaction():
        async with session.begin():
            yield
    else:
        yield
        if session.in_transaction():
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Create async sessionmaker
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=async_engine,
    class_=AsyncSession,
)

Base = declarative_base()


# Dependency to get async session
async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
