from evaluate import load

from src.utils.utils import convert_dictOfLists_to_listOfDicts, get_average


class Scorer(object):
    def __init__(self, evaluation_config, metrics):
        self.evaluation_config = evaluation_config
        self.metrics_toCompute = {"accuracy": False, "squad": False}

        if "Accuracy" in metrics:
            self.metrics_toCompute["accuracy"] = True
            self.accuracy_metric = load("accuracy")

        if "Squad" in metrics:
            self.metrics_toCompute["squad"] = True
            self.squad_metric = load("squad")

    def add_batch(self, batchOf_evalInfo):
        """
        Add batch to scorer

        Args:
            batchOf_evalInfo:

        Returns:

        """
        if self.metrics_toCompute["accuracy"]:
            self.accuracy_metric.add_batch(
                predictions=batchOf_evalInfo["predicted_choice"],
                references=batchOf_evalInfo["lbl"],
            )

        # Have to format the answer correctly for record since record
        # also has an answer key which promptsource requires and cannot be overwritten
        if self.evaluation_config.get_datasetConfig().dataset == "record":
            converted_answers = convert_dictOfLists_to_listOfDicts(
                {
                    "text": batchOf_evalInfo["text"],
                    "answer_start": batchOf_evalInfo["answer_start"],
                }
            )
            for answer in converted_answers:
                answer["text"] = answer["text"]
                answer["answer_start"] = answer["answer_start"]
            batchOf_evalInfo["answers"] = converted_answers

        if self.metrics_toCompute["squad"]:
            self.squad_metric.add_batch(
                predictions=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "prediction_text": batchOf_evalInfo["prediction_text"],
                    }
                ),
                references=convert_dictOfLists_to_listOfDicts(
                    {
                        "id": batchOf_evalInfo["id"],
                        "answers": batchOf_evalInfo["answers"],
                    }
                ),
            )

    def get_score(self):
        score = {}

        if self.metrics_toCompute["accuracy"]:
            score.update(self.accuracy_metric.compute())

        if self.metrics_toCompute["squad"]:
            squad_metrics = self.squad_metric.compute()
            # Scale SQUAD metrics to be between 0 and 1
            for metric, value in squad_metrics.items():
                squad_metrics[metric] = value / 100
            score.update(squad_metrics)

        for key, value in score.items():
            score[key] = float("%.3f" % value)

        score["average"] = get_average(score.values())

        return score
