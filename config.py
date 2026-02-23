from pathlib import Path

import tomli_w
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource

CONFIG_PATH = Path("config.toml")


class TreeSearchConfig(BaseSettings):
    num_draft_nodes: int = 3
    debug_prob: float = 0.3
    epsilon: float = 0.3
    max_iterations: int = 10


class ExecConfig(BaseSettings):
    timeout: int = 3600
    enable_type_checking: bool = True
    max_type_check_attempts: int = 3


class CodeConfig(BaseSettings):
    model: str = "gpt-5-mini"
    model_temp: float = 1.0


class AgentConfig(BaseSettings):
    k_fold_validation: int = 1
    evaluation_metrics: list[str] | None = None
    code: CodeConfig = CodeConfig()


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARL_", env_nested_delimiter="__", toml_file=CONFIG_PATH
    )

    out_dir: str = "./out"
    treesearch: TreeSearchConfig = TreeSearchConfig()
    exec: ExecConfig = ExecConfig()
    agent: AgentConfig = AgentConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        toml_src = TomlConfigSettingsSource(settings_cls, CONFIG_PATH)
        return (
            env_settings,
            dotenv_settings,
            file_secret_settings,
            toml_src,
            init_settings,
        )


_CONFIG: Config | None = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = _load_config()
    return _CONFIG


def _load_config() -> Config:
    config = Config()
    if not CONFIG_PATH.exists():
        # write default config if file doesn't exist
        CONFIG_PATH.write_text(tomli_w.dumps(config.model_dump()))

    return config
