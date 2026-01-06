## Installation Instructions

1. Set up the environment
```sh
make setup
```

2. Launch the GUI
```sh
streamlit run src/app.py
```

On Windows, `streamlit` might not be available in the terminal.
By providing the path to the executable, you can still run it:<br/>
`.\env\Scripts\streamlit.exe run .\src\app.py`


## About 

This repository is experimental and intended to be used as a successor to the **math-sdk** repo. Currently this repository is independent of the math-sdk, though does make use of the LookUpTable file outoput.


## Usage

Run the application from the **convex-optimizer** directory using: `https://github.com/StakeEngine/convex-optimizer.git`
By default, unoptimized LookUpTable files can be placed within `input_dir/`. The corresponding 'segmented' file is required to identify simulation criteria.


## Optimizer Details

The contrained optimization step is preformed using an implementation of [CVXPY](https://www.cvxpy.org/).
