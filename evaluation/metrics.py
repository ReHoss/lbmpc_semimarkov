"""
Metrics for evaluation new algorithm
"""

from barl.util.misc_util import Dumper

class MetricWizard():
    def __init__(self, dumper:Dumper) -> None:
        self.dumper = dumper
        
        self.eig_evolution = []

    def register_eig_evolution(self, max_eig:float, old_eig:list, new_eig:list, selected_sample:list, evaluated_samples:list) -> None:
        """
        Register the evolution of expected information gain between two runs
        :param float max_eig: maximum eig value of new run
        :param list old_eig: original eig values
        :param list new_eig: eig values of new run
        :param list selected_sample: sample selected on previous run
        :param list evaluated_samples: samples currently evaluated
        """
        # Save samples and expected information gain
        self.dumper.add("Old Samples", evaluated_samples, verbose=False)
        self.dumper.add("Old EIG values", old_eig, verbose=False)
        self.dumper.add("New EIG values", new_eig, verbose=False)
        # Calculate metrics
        differences_from_last_run = []
        differences_from_max = []
        distance_to_max_point = []
        for point in range(len(old_eig)):
            differences_from_last_run.append(self.get_eig_evolution_single(old_eig[point], new_eig[point]))
            differences_from_max.append(self.get_eig_evolution_single(max_eig, new_eig[point]))
            distance_to_max_point.append(self.observe_distance(selected_sample, evaluated_samples[point]))
        
        self.eig_evolution.append(differences_from_last_run) # keep in object
        self.dumper.add("EIG Evolution", differences_from_last_run, verbose=False)
        self.dumper.add("Distance to Selected Point", distance_to_max_point, verbose=False)
        self.dumper.add("EIG Loss to Max", differences_from_max, verbose=False)

    def get_eig_evolution_single(self, old_eig:float, new_eig:float) -> None:
        """
        Get the evolution of expected information gain between two runs for a single point
        :param float old_eig: original eig value
        :param float new_eig: eig value of new run
        """
        return new_eig - old_eig

    def observe_distance(self, selected_sample:list, evaluated_sample:list) -> float:
        """
        Calculate distance of points
        :param list selected_sample: sample selected on previous run
        :param list evaluated_sample: sample currently evaluated
        :returns float: distance
        """
        distance = []
        for i in range(len(selected_sample)):
            distance.append(evaluated_sample[i] - selected_sample[i])
        return distance

    def get_eig_evolution(self) -> list:
        """
        Return evolution of eig
        :returns list: list of tuples with eig differences and distances from selected point
        """
        return self.eig_evolution
    