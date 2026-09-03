"""FastAPI 入口（路线文档 §13.1）。

启动：`uvicorn app.main:app --reload`

生产路径：lifespan 里按 `.env` 构造 gateway / graph / 数据库（`build_services`）。
测试路径：`create_app(services=...)` 注入 fake services，见 `tests/integration/test_api.py`。
"""

from app.api.app import create_app

app = create_app()
