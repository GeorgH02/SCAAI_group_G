# Local llama.cpp server

The launchers require a recent `llama-server` executable supporting `--jinja`
and `--gpu-layers auto`. They do not install software or run in the background.
They default to Qwen2.5 7B Instruct Q4_K_M, model alias `vienna-stay`, loopback
`127.0.0.1:8080`, 16,384 context tokens, one slot and automatic GPU-layer selection.
The first Hugging Face launch downloads weights into llama.cpp's cache. Allow
roughly 4.7 GB for these weights plus additional runtime memory. The repository
reference is not revision-pinned; record model hash and llama.cpp version for
reproducible benchmarks. Local GGUF paths support using a previously verified file.

## Windows

Install using the [official Windows instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md):

```powershell
winget install llama.cpp
```

Reopen PowerShell, change to the project folder, and run:

```powershell
.\scripts\start_llm.ps1
```

If your PowerShell policy blocks a downloaded script, review its contents first;
then, if permitted by your organization's policy, launch it for this process only:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_llm.ps1
```

Examples:

```powershell
.\scripts\start_llm.ps1 -GpuLayers 0
.\scripts\start_llm.ps1 -Server 'C:\Tools\llama.cpp\llama-server.exe'
.\scripts\start_llm.ps1 -ModelFile 'D:\Models\qwen.gguf' -ContextSize 8192
.\scripts\start_llm.ps1 -ModelRepo 'Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0'
.\scripts\start_llm.ps1 -DryRun
```

## Linux

Use an installed `llama-server` or follow the [official build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
For example, with Git, CMake, a C++ compiler and required development libraries
installed, build a CPU version outside the project:

```bash
git clone https://github.com/ggml-org/llama.cpp.git "$HOME/llama.cpp"
cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$HOME/llama.cpp/build" --config Release --target llama-server -j 4
```

From the project folder:

```bash
bash scripts/start_llm.sh --server "$HOME/llama.cpp/build/bin/llama-server"
```

If the executable is already on PATH, simply run `bash scripts/start_llm.sh`.
Examples:

```bash
bash scripts/start_llm.sh --gpu-layers 0
bash scripts/start_llm.sh --model-file '/opt/models/qwen.gguf' --context-size 8192
bash scripts/start_llm.sh --model-repo 'Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0'
bash scripts/start_llm.sh --dry-run
bash scripts/start_llm.sh --help
```

GPU offload requires a compatible llama.cpp build and drivers. See the official
build guide for CUDA/Vulkan options; selecting GPU layers cannot add a missing
backend. Use `0` for CPU, `auto` for automatic selection or a layer count for
partial offload. Lower the context size/layer count if memory is insufficient.

## Readiness and connection

Keep the launcher terminal open. In another terminal, check readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

Linux equivalent: `curl --fail http://127.0.0.1:8080/health`.
Loading may initially return HTTP 503; HTTP 200 with status `ok` means ready.
Launchers report their intended address, not confirmed readiness, and inherit the
server exit code. They use warning/error log verbosity, with no log file configured;
do not enable verbose prompt logging with private data. Use Ctrl+C to stop.

Connect from the project with `local_model(model='vienna-stay')`. If you change
the port, pass the matching `base_url='http://127.0.0.1:PORT/v1'` as well.
The server's web interface is available at <http://127.0.0.1:8080> when ready.
This launches inference only; the project's public chat application and online
guard classifiers remain pending. See [LangChain integration](langchain.md).

Flags and endpoints: [llama.cpp server reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
