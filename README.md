To add Sentence Transormer library without using torch or CUDA only if you use UV, or you can search for pip: 


Add these at the end of your pyproject.toml file: 

```
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
```

and run 

```
uv sync
```