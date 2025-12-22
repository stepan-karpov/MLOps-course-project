"""Скрипт для создания графиков метрик."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from hydra import compose, initialize_config_dir

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_metrics(run_id: Optional[str] = None) -> None:
    """
    Создание графиков метрик из MLflow.

    Args:
        run_id: ID запуска MLflow (если None, берется последний)
    """
    # Получаем корень проекта (абсолютный путь)
    project_root = Path(__file__).resolve().parent.parent.parent
    config_dir = project_root / "configs"

    # Инициализация Hydra с абсолютным путем к конфигам
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        client = mlflow.tracking.MlflowClient()

        # Получение последнего запуска
        if run_id is None:
            experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
            if experiment is None:
                print("Эксперимент не найден")
                return
            runs = client.search_runs(
                experiment.experiment_id, order_by=["start_time desc"], max_results=1
            )
            if not runs:
                print("Запуски не найдены")
                return
            run_id = runs[0].info.run_id

        # Получение истории метрик
        def get_metric_history(metric_name: str):
            """Получает историю метрики по эпохам."""
            try:
                history = client.get_metric_history(run_id, metric_name)
                if history:
                    steps = [m.step for m in history]
                    values = [m.value for m in history]
                    return steps, values
            except Exception:
                pass
            return None, None

        plots_dir = project_root / cfg.paths.plots_dir
        plots_dir.mkdir(parents=True, exist_ok=True)

        # График Loss
        train_loss_steps, train_loss_values = get_metric_history("train_loss")
        val_loss_steps, val_loss_values = get_metric_history("val_loss")

        if train_loss_steps is not None or val_loss_steps is not None:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if train_loss_steps is not None:
                ax.plot(train_loss_steps, train_loss_values, label="Train Loss", marker="o")
            if val_loss_steps is not None:
                ax.plot(val_loss_steps, val_loss_values, label="Val Loss", marker="s")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
            plt.close()

        # График Accuracy
        train_acc_steps, train_acc_values = get_metric_history("train_acc")
        val_acc_steps, val_acc_values = get_metric_history("val_acc")

        if train_acc_steps is not None or val_acc_steps is not None:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            if train_acc_steps is not None:
                ax.plot(train_acc_steps, train_acc_values, label="Train Acc", marker="o")
            if val_acc_steps is not None:
                ax.plot(val_acc_steps, val_acc_values, label="Val Acc", marker="s")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / "accuracy_curves.png", dpi=300, bbox_inches="tight")
            plt.close()

        # График F1
        train_f1_steps, train_f1_values = get_metric_history("train_f1")
        val_f1_steps, val_f1_values = get_metric_history("val_f1")

        if train_f1_steps is not None or val_f1_steps is not None:
            fig, ax = plt.subplots()
            ax.set_title("Training and Validation F1 Score")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("F1 Score")
            if train_f1_steps is not None:
                ax.plot(train_f1_steps, train_f1_values, label="Train F1", marker="o")
            if val_f1_steps is not None:
                ax.plot(val_f1_steps, val_f1_values, label="Val F1", marker="s")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(plots_dir / "f1_curves.png", dpi=300, bbox_inches="tight")
            plt.close()

        print(f"Графики сохранены в {plots_dir}")


if __name__ == "__main__":
    plot_metrics()
