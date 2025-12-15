
# Social-Suicide-ABM

An Agent-Based Model (ABM) of a small community, simulating the onset of suicidal thoughts with social effects.


## Modules
```
root/
├── model/
│   ├── agents/                    # Main app component
|   |   └── StandardAgent.py       # Default agent class with main agent action definitions
│   ├── parameters/
|   |   ├── sets/                  # Record classes to contain parameters per update equation
|   |   ├── Parameters.py          # Abstract parameters superclass with getters/setters for all equation sets
|   |   └── DefaultParameters.py   # Extension of Parameters that initializes with default values
│   ├── system_updates/            # Location state representations, AgentUpdater with evolution functions
│   └── SuicideModel.py            # Model class that initializes the environment
├── .gitignore
├── requirements.txt               # Python library requirements
└── run_model.py                   # Runs the model with presets
```
## Run Locally

Clone the project

```bash
  git clone https://github.com/MikkoBra/Social-Suicide-ABM
```

Go to the project directory

```bash
  cd Social-Suicide-ABM
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the model

```bash
  python run_model.py
```


## Authors

- [Mikko Brandon](https://www.github.com/MikkoBra)

