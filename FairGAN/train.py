import argparse
import pandas as pd
from DataProcessor import DataProcessor
from DataProcessor import DatasetPipeline
from Config import Config
import tensorflow_ranking as tfr
import tensorflow_probability as tfp
from Model import FairGAN
import numpy as np


if __name__ == "__main__":
    ds_name = "ebnerd"

    print("Loading dataset {}..." .format(ds_name))
    train_df = pd.read_csv(f"data/{ds_name}/train_df.csv")
    valid_df = pd.read_csv(f"data/{ds_name}/valid_df.csv")
    test_df = pd.read_csv(f"data/{ds_name}/test_df.csv")

    train_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
    valid_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
    test_df.rename(columns={"user_id": "user", "item_id": "item"}, inplace=True)
    train_df['rating'] = 1
    valid_df['rating'] = 1
    test_df['rating'] = 1

    ratings = pd.concat([train_df, valid_df, test_df], axis=0)

    train = DataProcessor.construct_one_valued_matrix(ratings, train_df, item_based=False)
    valid = DataProcessor.construct_one_valued_matrix(ratings, valid_df, item_based=False)
    test = DataProcessor.construct_one_valued_matrix(ratings, test_df, item_based=False)

    train_ds = DatasetPipeline(labels=train.toarray(), conditions=train.toarray()).shuffle(1)
    valid_ds = DatasetPipeline(labels=valid.toarray(), conditions=train.toarray()).shuffle(1)
    test_ds = DatasetPipeline(labels=test.toarray(), conditions=train.toarray()).shuffle(1)

    config = Config
    config['n_items'] = train.shape[1]
    # Metrics
    metrics = [
        # Precision
        tfr.keras.metrics.PrecisionMetric(topn=5, name="P@5"),
        tfr.keras.metrics.PrecisionMetric(topn=10, name="P@10"),
        tfr.keras.metrics.PrecisionMetric(topn=15, name="P@15"),
        tfr.keras.metrics.PrecisionMetric(topn=20, name="P@20"),

        # Recall
        tfr.keras.metrics.RecallMetric(topn=5, name="R@5"),
        tfr.keras.metrics.RecallMetric(topn=10, name="R@10"),
        tfr.keras.metrics.RecallMetric(topn=15, name="R@15"),
        tfr.keras.metrics.RecallMetric(topn=20, name="R@20"),

        # NDCG
        tfr.keras.metrics.NDCGMetric(topn=5, name="G@5"),
        tfr.keras.metrics.NDCGMetric(topn=10, name="G@10"),
        tfr.keras.metrics.NDCGMetric(topn=15, name="G@15"),
        tfr.keras.metrics.NDCGMetric(topn=20, name="G@20"),
    ]

    best_params = None
    best_evals = None
    best_score = 0.0

    for lr in [1e-6, 1e-5, 1e-4]:
        for batch in [64, 128]:
            for epochs in [50, 100]:

                config['ranker_gen_lr'] = lr
                config['ranker_dis_lr'] = lr
                config['batch'] = batch
                config['epochs'] = epochs

                model = FairGAN(metrics,**config)

                model.fit(train_ds.shuffle(train.shape[0]).batch(config['batch'], True), epochs=config['epochs'], callbacks=[])

                evals = model.evaluate(valid_ds.batch(train.shape[0]))
                score = np.mean(evals)
                if score > best_score:
                    best_score = score
                    best_params = (lr, batch, epochs)
                    best_evals = evals

                del model

    print("Best parameters: lr={}, batch={}, epochs={}".format(*best_params))
    print("Best evaluation: ", best_evals)
    print("Best score: ", best_score)

    # Evaluate on test set

    train_with_valid = train + valid
    train_with_valid_ds = DatasetPipeline(labels=train_with_valid.toarray(), conditions=train_with_valid.toarray()).shuffle(1)
    
    config['ranker_gen_lr'] = best_params[0]
    config['ranker_dis_lr'] = best_params[0]
    config['batch'] = best_params[1]
    config['epochs'] = best_params[2]

    model = FairGAN(metrics,**config)

    model.fit(train_with_valid_ds.shuffle(train_with_valid.shape[0]).batch(config['batch'], True), epochs=config['epochs'], callbacks=[])

    evals = model.evaluate(test_ds.batch(train_with_valid.shape[0]))
    print("\nEvaluate on test set:")
    print(evals)
    print("Score: ", np.mean(evals))

    with open(f"FairGAN_{ds_name}_results.txt", "w") as f:
        f.write("Best parameters: lr={}, batch={}, epochs={}\n".format(*best_params))
        f.write("Best evaluation: {}\n".format(best_evals))
        f.write("Best score: {}\n".format(best_score))
        f.write("\nEvaluate on test set:\n")
        f.write(str(evals))
        f.write("\nScore: {}\n".format(np.mean(evals)))