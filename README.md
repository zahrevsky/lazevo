# Lazevo — Lagrangian Zeldovich Void Finder<br><sup>Helps you find cosmological voids without building density maps<br/></sup>

<br/>

## How to run
Before first run, install dependencies:
```
pipenv install
```
Now, to run Lazevo, use this command:
```
pipenv run python src/main.py
```
This will run Lazevo on data from `input/` directory, using predefined config in `input/params.yaml`.


## Documentation
[PIZA algorithm implementation & data structures](docs/piza.md)

[Plotting routines](docs/plotter.md)

[Miscelaneous utilities](docs/utils.md)