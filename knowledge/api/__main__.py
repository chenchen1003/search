from knowledge.config import settings
from knowledge.api.server import run

run(chroma_dir=settings.chroma_dir, embed_model=settings.embed_model, port=settings.api_port)
