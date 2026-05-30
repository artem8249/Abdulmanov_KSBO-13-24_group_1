from __future__ import annotations

import typer
import uvicorn

from src.utils.config import load_config
config = load_config()

app = typer.Typer(help="CLI для запуска HTTP-сервиса Рекомендательной Системы H&M")

@app.command()
def serve(
    host: str = typer.Option(config["api"]["host"], help="Хост для запуска сервера."),
    port: int = typer.Option(config["api"]["port"], help="Порт для запуска сервера."),
) -> None:
    typer.echo(f"Запуск сервера на http://{host}:{port}")
    typer.echo(f"Документация Swagger: http://{host}:{port}/docs")
    
    uvicorn.run("src.service.api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    app()