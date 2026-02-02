# Creating New UV Project

If you want to create a new project, first decide the python version you will be using (3.12 is a good bet). Next, decide where *you* want to install the virtual environment. The project will always consider `setup/.venv` as the location by default. There is never a good reason to change this, but if you must, it can be done by altering the [config file](../configs/private_vars.yaml). 

First, create the environment directory wherever you want it, and then symlink it to the `setup/.venv` directory
```console
# Run this in root directory of the project
uv venv /path/to/venv --python=3.XX
ln -s /path/to/venv setup/.venv 
```
If you are doing this, make sure to set the UV_CACHE environment variable to the same filesystem as the environment.


cd into the [setup](../setup) directory and then run:
```console
uv init
rm main.py
```

We never use this main file, so delete it. This will create a project with the name `setup`. To change the project name, edit the `pyproject.toml` file. Also, make sure to add the exact python version to `pyproject.toml`, i.e. make it `==` not `>=` and then open `.python-version` and add the exact python version there as well.

You can now add pip packages with:
```console
uv add click huggingface_hub seaborn
```
These three packages are required for this codebase to run, but you can remove the `huggingface_hub` dependency by removing [`sync_data.py`](../sync_data.py) and remove the `seaborn` dependency by removing [Plotter](../utils/plot_handling.py).

This creates a uv.lock file, that must be committed to version control. This will also create the environment. Source it with (from root):

```console
source setup/.venv/bin/activate
```

# Instruction For Project Reproduction (Replace 3.XX with version and put in README)

This project uses Python 3.XX with [uv](https://docs.astral.sh/uv/guides/projects/) to manage dependencies, but there is one significant design decision: the 'uv project' is in the `setup` directory, as opposed to root. This means that once you've created the environment, you will find the virtual environment in `setup/.venv/`. 

First, ensure you have installed uv in your python.
```console
pip install --upgrade pip  uv
```

## Installing to alternate locations

Next, decide where you want to store the virtual environment. Some users may *not* want to install the environment into `setup/.venv/` (perhaps the filesystem space is limited and you want the env to be elsewhere). If you're fine saving in `setup/.venv/` directly, skip to the next section. 

If you want to install to an alternate location, first create the environment elsewhere and create a symbolic link to the `setup/.venv` directory. 
```console
# Run this in root directory of the project
uv venv /path/to/venv --python=3.XX
ln -s /path/to/venv setup/.venv 
```
If you are doing this, make sure to set the UV_CACHE environment variable to the same filesystem as the environment.

## Installation
Finally, navigate to the setup folder and run `uv sync`:

```console
cd setup
uv sync
```

This will create a virtual environment in setup/.venv. Before running any code in this repo, make sure to run (from root):

```console
source setup/.venv/bin/activate
```
