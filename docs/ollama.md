# Local Ollama server

The project uses `qwen2.5:7b` through Ollama's OpenAI-compatible API at
`http://127.0.0.1:21434/v1`. No llama.cpp build or host Ollama install is required.
Install Docker with Compose and run from the project folder:

```bash
bash scripts/start_llm.sh
```

On Windows:

```powershell
.\scripts\start_llm.ps1
```

The launchers start only the Ollama service, wait up to 120 seconds for its API,
then pull the model. A failed startup or pull stops the script with an error.
The first download is about 4.7 GB; subsequent pulls reuse existing layers.
Ollama keeps running in Docker after the script exits. The `ollama_data` volume
persists weights across container restarts.

Equivalent manual commands:

```bash
docker compose up -d --wait --wait-timeout 120 ollama
docker compose exec -T ollama ollama pull qwen2.5:7b
```

Preview without starting services or downloading weights:

```bash
bash scripts/start_llm.sh --dry-run
```

PowerShell equivalent: `.\scripts\start_llm.ps1 -DryRun`.
For another tool-capable model, pass `--model MODEL` (Bash) or `-Model MODEL`
(PowerShell), then use the same name in `local_model(model='MODEL')`.
The previous llama.cpp launcher flags for GGUF files, GPU layers and ports
have been replaced by Ollama model names and settings in `compose.yaml`.

## Readiness and inference

```bash
docker compose exec -T ollama ollama list
curl --fail http://127.0.0.1:21434/api/tags
```

In PowerShell, use `Invoke-RestMethod http://127.0.0.1:21434/api/tags`.
Check that `qwen2.5:7b` appears; container health alone does not mean the model
has been downloaded or loaded into memory. Loading happens at first inference.

```python
from langchain_agent.workflow import local_model
model = local_model()
print(model.invoke('Say hello in one sentence.').content)
```

Run [try_agent.ipynb](../try_agent.ipynb) to exercise tool calling with fictional
listings. Its saved outputs are cleared so a previous backend's results are not
presented as a successful Ollama run.

## Configuration and shutdown

Compose binds the host port to `127.0.0.1:21434`, uses CPU inference, a
16,384-token context and one parallel request. Change `OLLAMA_CONTEXT_LENGTH`
in `compose.yaml` and recreate the service to change context size; it cannot be
set through the OpenAI API. GPU acceleration requires additional driver and
Docker configuration. The image uses `latest`; pin a tested version or digest
when reproducibility matters.

The project uses host port 21434 so it can run alongside a host Ollama
installation on port 11434. Inside Docker, Ollama still listens on port
11434. The data pipeline starts PostgreSQL and Qdrant independently.

Stop inference with `docker compose stop ollama`. `docker compose down` stops
all project services while preserving volumes. `docker compose down -v` also
deletes stored database data and model weights.

References: [Ollama Docker setup](https://docs.ollama.com/docker),
[OpenAI compatibility and tools](https://docs.ollama.com/api/openai-compatibility),
[context length](https://docs.ollama.com/context-length).
