# RF-PHATE

RF-PHATE is a package which allows the user to create random forest-based supervised, low-dimensional embeddings based on the 
manifold learning algorithm described in 
[Random Forest-Based Diffusion Information Geometry for Supervised Visualization and Data Exploration](https://ieeexplore.ieee.org/document/9513749)

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Toolbox like below. 
Rerun this command to check for and install  updates .
```bash
pip install git+https://github.com/jakerhodes/rfphate
```

## Usage
Features:
* functions.listChunker  --> generator that chunks and interable in evenly sized chunks 
* functions.weirdCase    --> converts a string to a totally unreadable format
* functions.report      --> prints to the console with a timestamp
* decorators.singleton  --> used for decoratint your class to make it a singleton

#### Demo of some of the features:
```python
from rfphate import MakeRFPHATE
from dataset import normalize_data
import pandas as pd
import seaborn as sns

# Read in the data
data   = pd.read_csv('../data/auto-mpg.csv', sep = ',')
x, y   = dataset.normalize_data(data, label_col = 0)

rfphate_op = MakeRFPHATE(label_type = 'numeric)
embedding = rfphate_op.fit_transform(x, y)

sns.scatterplot(x = emb[:, 0], y = emb[:, 1], hue = data.iloc[:, 0])
```



## License
[GNU-3](https://www.gnu.org/licenses/gpl-3.0.en.html)
