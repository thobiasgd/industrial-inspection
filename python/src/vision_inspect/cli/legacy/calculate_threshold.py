from vision_inspect.config import PROJECT_ROOT

import torch


def main() -> None:
    project_root = PROJECT_ROOT
    scores_path = project_root / "artifacts" / "normal_scores.pt"

    if not scores_path.is_file():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    data = torch.load(
        scores_path,
        map_location="cpu",
        weights_only=True,
    )

    normal_scores = data["scores"]

    percentile_95 = torch.quantile(normal_scores, 0.95).item()
    percentile_99 = torch.quantile(normal_scores, 0.99).item()
    maximum = normal_scores.max().item()

    threshold = percentile_99

    print("Number of normal images:", len(normal_scores))
    print(f"95th percentile: {percentile_95:.4f}")
    print(f"99th percentile: {percentile_99:.4f}")
    print(f"Maximum score: {maximum:.4f}")
    print(f"Selected threshold: {threshold:.4f}")

    threshold_path = project_root / "artifacts" / "threshold.pt"

    torch.save(
        {
            "threshold": threshold,
            "method": "99th_percentile",
        },
        threshold_path,
    )

    print("Saved to:", threshold_path)


if __name__ == "__main__":
    main()
