from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8000

    aoai_deploy_embed_3_large: str
    aoai_deploy_embed_3_small: str
    aoai_api_key: str
    aoai_endpoint: str
    aoai_deploy_embed_ada: str

    aoai_deploy_gpt4o: str
    aoai_deploy_gpt4o_mini: str
    aoai_endpoint: str

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()