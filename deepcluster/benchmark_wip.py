import argparse
from clustpy.data import load_mnist, load_cifar10
from clustpy.deep import DEC, DCN, IDEC, VaDE
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, silhouette_score as silhouette
from clustpy.utils import EvaluationAlgorithm, evaluate_dataset, EvaluationMetric
import time
import subprocess

# utils -> evaluate

DATASET_LOADERS = {
    'mnist': load_mnist,
    'cifar10': load_cifar10
}

ALGORITHMS = {
    'dec': DEC,
    'dcn': DCN,
    'vade': VaDE
}


def create_small_dataset(data, labels, sample_size=1000):
    """Create a smaller dataset for testing."""
    indices = np.random.choice(len(data), sample_size, replace=False)
    return data[indices], labels[indices]


def main(args):
    data_loader = DATASET_LOADERS[args.dataset]
    data, labels = data_loader()

    # testing purpose
    data, labels = create_small_dataset(data, labels)
    n_clusters = len(np.unique(labels))


    # benchmarking clustering algorithm
    algorithm1 = ALGORITHMS[args.algorithm1]
    algorithm2 = ALGORITHMS[args.algorithm2]

    n_repetitions = 2
    aggregations = [np.mean, np.std, np.max]

    algorithms = [
        EvaluationAlgorithm(name=args.algorithm1, algorithm=algorithm1, params={"n_clusters": None}),
        EvaluationAlgorithm(name=args.algorithm2, algorithm=algorithm2, params={"n_clusters": None})
    ]

    metrics = [EvaluationMetric(name="nmi", metric=nmi, params={"average_method": "geometric"}, use_gt=True),
               EvaluationMetric(name="silhouette", metric=silhouette, use_gt=False)]

    #df = evaluate_dataset(X=data, evaluation_algorithms=algorithms, evaluation_metrics=metrics, labels_true=labels,
    #                      n_repetitions=n_repetitions, aggregation_functions=aggregations, random_state=1, add_n_clusters=True)

    #df.to_excel('test.xlsx')

    start_time = time.time()

    subprocess.run(['python', 'train_deep_cluster.py'])

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # benchmarking classification
    # model = algo.algorithm(n_clusters=n_clusters)
    # model.fit(data)
    # embeddings = model.predict(data)  # Ensure predict method returns embeddings
    # pseudo_labels = model.fit_predict(data)
    #
    # # Split data for classification
    # train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(embeddings, labels, test_size=0.2,
    #                                                                                 random_state=42)
    #
    # # Train classifier with pseudo-labels
    # acc_pseudo, prec_pseudo, rec_pseudo, f1_pseudo = train_classifier_with_pseudo_labels(train_embeddings,
    #                                                                                      pseudo_labels, test_embeddings,
    #                                                                                      test_labels)
    #
    # # Train classifier with real labels
    # acc_real, prec_real, rec_real, f1_real = train_classifier_with_real_labels(train_embeddings, train_labels,
    #                                                                            test_embeddings, test_labels)
    #
    # print(
    #     f"{algo.name} - Classifier with Pseudo-Labels: Accuracy={acc_pseudo}, Precision={prec_pseudo}, Recall={rec_pseudo}, F1-Score={f1_pseudo}")
    # print(
    #     f"Classifier with Real Labels: Accuracy={acc_real}, Precision={prec_real}, Recall={rec_real}, F1-Score={f1_real}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark deep clustering algorithms.')
    parser.add_argument('--dataset', type=str, choices=DATASET_LOADERS.keys(), required=True,
                        help='The dataset to use for benchmarking.')
    parser.add_argument('--algorithm1', type=str, choices=ALGORITHMS.keys(), required=True,
                        help='The algorithm which we compare with.')
    parser.add_argument('--algorithm2', type=str, choices=ALGORITHMS.keys(), required=True,
                        help='The algorithm which we compare with.')

    args = parser.parse_args()
    main(args)
