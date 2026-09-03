galvani
=======

<!---
SPDX-FileCopyrightText: 2013-2020 Christopher Kerr, Peter Attia

SPDX-License-Identifier: GPL-3.0-or-later
-->

Read proprietary file formats from electrochemical test stations.

# Usage

## Bio-Logic .mpr files

Use the `MPRfile` class from BioLogic.py (exported in the main package)

```python
from galvani import BioLogic
import pandas as pd

mpr_file = BioLogic.MPRfile('test.mpr')
df = pd.DataFrame(mpr_file.data)
```

## Arbin .res files

Use the `./galvani/res2sqlite.py` script to convert the .res file to a sqlite3 database with the same schema, which can then be interrogated with external tools or directly in Python.
For example, to extract the data into a pandas DataFrame (will need to be installed separately):

```python
import sqlite3
import pandas as pd
from galvani.res2sqlite import convert_arbin_to_sqlite
convert_arbin_to_sqlite("input.res", "output.sqlite")
with sqlite3.connect("output.sqlite") as db:
    df = pd.read_sql(sql="select * from Channel_Normal_Table", con=db)
```

This functionality requires [MDBTools](https://github.com/mdbtools/mdbtools) to be installed on the local system.

## Metrohm AUTOLAB NOVA .nox files

Use the `NOXfile` class to read calculated signals from NOVA files. This is a
pure-Python reader: NOVA, the Metrohm SDK, and a .NET runtime are not required.

```python
from galvani import NOXfile

nox = NOXfile("cycling.nox")

# A combined NumPy record array for time/potential/current datasets
time = nox.data["time/s"]
potential = nox.data["Ewe/V"]
current = nox.data["I/A"]

# Or inspect every recorded command and its native NOVA signal names
for dataset in nox.datasets:
    print(dataset.command, dataset.signal_names)
```

# Installation

The latest galvani releases can be installed from [PyPI](https://pypi.org/project/galvani/) via

```shell
pip install galvani
```

The latest development version can be installed with `pip` directly from GitHub:

```shell
pip install git+https://codeberg.org/echemdata/galvani
```

## Development installation and contributing 

If you wish to contribute to galvani, please clone the repository and install the testing dependencies:

```shell
git clone git@codeberg.org:echemdata/galvani
cd galvani
pip install -e .\[tests\]
```

Code can be contributed back via [pull requests](https://codeberg.org/echemdata/galvani/pulls) and new features or bugs can be discussed in the [issue tracker](https://codeberg.org/echemdata/galvani/issues).
