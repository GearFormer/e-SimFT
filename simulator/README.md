# gear_train_simulator
A module to simulate and analyze a gear train design

<h3>Requirements</h3>

1. Recommended python 3.7 or higher.

2. Install dymos.
```
pip install dymos
```
* (You might have to install ```packaging``` before dymos)

<h3>Running examples</h3>

``` 
python gear_train_simulator.py test/succeeds/ex1_1.json
```

<h3>Simulating multiple gear trains in parallel</h3>

```
python generate.py
```
Replace the ```input_data``` in the script with a list of randomly generated gear sequence + input parameters.

Change ```num_threads``` based on your machine.
