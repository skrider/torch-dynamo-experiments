from . import bert
from . import simple

experiment_dict = {
    "bert": bert.make_experiment,
    "simple": simple.make_experiment
}
