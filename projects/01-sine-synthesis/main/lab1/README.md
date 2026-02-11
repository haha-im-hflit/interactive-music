# Lab 1 setup (Sine Generator)

`imslib` for this course is **not** published on PyPI as a normal pip package.

This lab expects `imslib` to exist in:

`projects/01-sine-synthesis/main/imslib`

because `main.py` prepends `..` to `sys.path`.

## Quick setup

From this folder, run:

```bash
bash download_imslib.sh
```

That script downloads `unit1/imslib` from the IMS course repo and places it in
the expected location.

## Run

```bash
python3 main.py
```
