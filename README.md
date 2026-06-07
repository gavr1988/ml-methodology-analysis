# ML Data Analysis Setup

This repository is being prepared to analyze the Fashion MNIST dataset with Python.

## What is happening

We created a virtual environment at `.venv` and installed the packages that `run.py` imports:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `tensorflow`

We also generated a `requirements.txt` file so the same environment can be recreated later.

## Why we are doing this

Using a virtual environment keeps the project dependencies isolated from the rest of your computer. That makes the project easier to reproduce, easier to share, and less likely to break because of unrelated Python packages installed elsewhere.

The TensorFlow install is needed because `run.py` imports `tensorflow.keras.datasets.fashion_mnist` to load the Fashion MNIST data.

## Why VS Code showed an import error

The code runs successfully when executed with the `.venv` Python interpreter, but Pylance in VS Code can still show `Import "tensorflow.keras.datasets" could not be resolved` if the editor is using a different interpreter.

That warning is about editor configuration, not necessarily a broken install.

## How to run the script

Activate the virtual environment first:

```bash
source .venv/bin/activate
```

Then run the script:

```bash
python run.py
```

If you want to bypass activation, run it directly:

```bash
.venv/bin/python run.py
```

## If the editor still shows warnings

Make sure VS Code is using the project venv interpreter:

1. Open the Command Palette.
2. Run `Python: Select Interpreter`.
3. Choose `.venv/bin/python`.
4. Reload the window if the warning remains.

## Project files

- `run.py` contains the analysis starter code.
- `requirements.txt` records the installed packages.
- `.gitignore` keeps local environment files out of version control.
