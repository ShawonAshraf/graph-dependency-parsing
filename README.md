# graph-dependency-parsing


## parser information
- decoder : CLE

## running the code
### setting up the env
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### entry point
```bash
cd src
python main.py --mode --trainf --devf --testf
```

--trainf, --devf, --testf are the paths to the training,
dev and test files.

### hparams
Currently the model trains for `50` epochs

### loading models
`Perceptron` class has a `load` method which can be used to load pretrained weights
stored as pickles.

```python
perceptron = Perceptron()
perceptron.load(file_path)
```

If `is_train` (default True) is marked as false, weights will have
to be loaded manually. For the default case, the model will initialize weights when
`perceptron.batchify_features(sentences)` is called.
