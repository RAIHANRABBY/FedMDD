from flwr.server.strategy import FedAvg
from task import load_model
import flwr as fl
from typing import List, Tuple, Dict
from flwr.common import Metrics

# Custom function to aggregate accuracy from clients
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate a weighted average of accuracy based on client data sizes."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    examples = [num_examples for num_examples, m in metrics if "accuracy" in m]

    # Return weighted average accuracy
    return {"accuracy": sum(accuracies) / sum(examples) if examples else 0}



# def evaluate_fn(parameters: fl.common.Parameters) -> Tuple[float, Dict[str, float]]:
#     model = load_model()
#     model.set_weights(fl.common.parameters_to_ndarrays(parameters))
#     # Perform evaluation on server-side dataset here
#     loss, accuracy = model.evaluate(server_x, server_y)
#     return loss, {"accuracy": accuracy}

def start_server():
    # Load model and set initial weights
    model = load_model()
    initial_parameters = model.get_weights()

    # Define strategy with metric aggregation functions
    strategy = FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        #  evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregates accuracy
    )

    # Start Flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
