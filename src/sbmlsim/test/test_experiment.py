from sbmlsim.examples.experiments.glucose import glucose
from sbmlsim.examples.experiments.repressilator import repressilator


def test_repressilator_experiment(tmp_path):
    repressilator.run(output_path=tmp_path)


def test_glucose_experiment(tmp_path):
    glucose.glucose_experiment()
