from .benchmark import run_benchmark, summarize


def main():
    rows = run_benchmark(
        train_seeds=[100, 101, 102],
        test_seeds=[200, 201, 202],
        n_jobs=5,
        n_machines=3,
        operations_per_job=3,
        time_limit_seconds=0.25,
    )
    summary = summarize(rows)
    print("controller,mean_makespan,mean_weighted_tardiness")
    for controller, metrics in summary.items():
        print(
            f"{controller},{metrics['mean_makespan']:.3f},"
            f"{metrics['mean_weighted_tardiness']:.3f}"
        )


if __name__ == "__main__":
    main()
