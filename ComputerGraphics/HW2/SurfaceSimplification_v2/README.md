## Quadric Error Metric (QEM)

### Dependencies

- Python 3.9
- numpy
- scipy
- polyscope

Install all the dependencies using the command following:

```shell
pip install -r requirements.txt
```

### Test

Enter the command below to test the QEM algorithm implemented in Python:

```shell
python main.py -i models/bunny.obj -r 0.5 -v
# -i|--input: input OBJ file path
# -r|--ratio: the ratio of simplification (0,1)
# -v|--visualization: to visualize
```

### Evaluation

1. Firstly, you should add `-s` in the command to save the simplified 3D mesh model as local OBJ file

   ```shell
   # example
   python main.py -i models/bunny.obj -r 0.5 -v -s
   # There will be a new OBJ file in `models/`
   ```

2. Modify the `obj_path1` and `obj_path2` in the `qem_utils.py` file, and run this script:

   ```shell
   python qem_utils.py
   ```

