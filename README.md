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


## Optimizer Details

The contrained optimization step is preformed using an implementation of [CVXPY](https://www.cvxpy.org/).
