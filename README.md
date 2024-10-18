This is the repository for the multi-armed coreset selector (MACES) method, conceived for my master's thesis project at ETH Zürich.

Here are examples of how to do a few things:

### Reproducing results from the manuscript

Running the files in the `Results_Scripts` folder will give you the results presented in the manuscript. E.g., 

```
os.chdir(r"...\GitHub\Master-Thesis")
%run results-metadata-in-model.py
```

Results are saved into the `Comprehensive Benchmarks` folder under the subfolder name specified in the results file.

Note that you will need to tweak the filepaths first in the `benchmarking_utils` and `results-...` scripts.

### Selecting a coreset with the `MACES` method

Let's suppose you want to run the MACES method to obtain a coreset with a pre-specified hidden, test and validation split. Then you can do:

```
from MACES_and_Benchmark_Methods.MACES import lazy_bandit

lazy_bandit_instance = lazy_bandit(dataset, ...) # the doc-string will help with the args
lazy_bandit_instance.run_MACES() # takes time...

# retrieve coreset as indices in the dataset
coreset_indices      = lazy_bandit_instance.train_indices
coreset              = dataset[dataset.index.isin(coreset_indices)]
```
