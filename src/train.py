"""Command-line training: python -m src.train --head-epochs 15 --ft-epochs 15

Mirrors the notebook so the model can be rebuilt from the repository alone.
"""
import argparse

from src import config, database as db
from src.model import evaluate, save_artifacts, train_new_model
from src.preprocessing import compute_class_weights, count_images, list_classes, make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=15)
    args = ap.parse_args()

    class_names = list_classes(config.TRAIN_DIR)
    print("Classes:", class_names)
    train_ds = make_dataset(config.TRAIN_DIR, class_names, 0.2, "training")
    val_ds = make_dataset(config.TRAIN_DIR, class_names, 0.2, "validation")
    test_ds = make_dataset(config.TEST_DIR, class_names, shuffle=False)
    weights = compute_class_weights(config.TRAIN_DIR, class_names)

    model, history = train_new_model(train_ds, val_ds, len(class_names), weights,
                                     args.head_epochs, args.ft_epochs)
    metrics = evaluate(model, test_ds, class_names)
    n_train = sum(count_images(config.TRAIN_DIR).values())
    save_artifacts(model, class_names, metrics, history, version=1, trigger="initial", n_train=n_train)

    db.init_db()
    db.add_model_version(1, "initial", n_train, 0, metrics["accuracy"], metrics["f1_macro"],
                         None, True, "initial training")
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
