from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
import os

# Use an async-compatible database URL, for example for MySQL:
# mysql+aiomysql://<user>:<password>@<host>/<dbname>
SQLALCHEMY_DATABASE_URL = f"mysql+aiomysql://{os.getenv('DATABASE_USER')}:{os.getenv('DATABASE_PASSWORD')}@{os.getenv('DATABASE_HOST')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"

# Create async engine
async_engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=True)

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
