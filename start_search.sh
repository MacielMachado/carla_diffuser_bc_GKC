#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

if [ "$1" == "-t" ]; then
    script_name="search_hyperparameters_t.py"
    screen_name="carla_bc_t"
else
    script_name="search_hyperparameters.py"
    screen_name="carla_bc"
fi

screen -L -S "$screen_name" .venv/bin/python "$script_name"
