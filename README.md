<h1>Smarter Mobility Data Challenge</h1>
This repository contains the code for forecasting occupation of charging stations in Paris, in order for utilities to optimize their production units in accordance with charging needs.

<br>
<h3>Installation</h3>
To install the necessary dependencies, create a virtual environment and activate it:

    pip install virtualenv

Create a new virtual environment by running the following command:

    virtualenv codalab_env
This will create a new virtual environment named myenv in the current directory. You can specify a different name if you want.

Activate the virtual environment by running the following command:

    source codalab_env/bin/activate

Once the virtual environment is activated, you should see the name of the environment in parentheses at the beginning of the command prompt, like this:

    (codalab_env) $

You can now install the necessary packages and run the code in the virtual environment.
 Then, run the following command:

    pip install -r requirements.txt

This will install all the required packages listed in the `requirements.txt` file.

<h3>Code style</h3>

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)


<h3>Usage</h3>
To run the charging status forecast task, use the following command:

    python src/main.py
This will execute the code in the `src/main.py` file and generate the charging status forecast.

<h3>Notes</h3>

You may need to modify the `src/main.py` file to specify the input data and parameters for the task.
The charging status forecast results will be saved in the `output` directory.
