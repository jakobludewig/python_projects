from metaflow import FlowSpec, step, Parameter

# see here: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


numeric_types = [
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
]


class WelfordSimFlow(FlowSpec):
    random_seed = Parameter(
        "random_seed",
        help="The random seed for sampling the datasets from Open ML as well as the columns from each dataset",
        default=93840235,
    )

    num_datasets = Parameter(
        "num_datasets",
        help="Number of Open ML datasets to sample for the simulation",
        default=10,
    )

    max_cols = Parameter(
        "max_cols",
        help="Maximum number of columns to calculate variances for in each dataset",
        default=10,
    )

    max_cols_large_datasets = Parameter(
        "max_cols_large_datasets",
        help="Maximum number of columns to calculate variances for in each dataset, applies to datasets with many observations (see parameter 'n_def_large_datasets')",
        default=3,
    )

    n_def_large_datasets = Parameter(
        "n_def_large_datasets",
        help="Threshold after which to apply the 'max_cols_large_datasets' parameter instead of 'max_cols'",
        default=1e5,
    )

    max_obs = Parameter(
        "max_obs",
        help="Maximum number of rows in a dataset allowed (will exclude datasets with more rows from analysis)",
        default=1e6,
    )

    @step
    def start(self):

        print(
            "Simulation started with seed {0} and {1} datasets to include in simulation.".format(
                self.random_seed, self.num_datasets
            )
        )

        self.next(self.sample_datasets)

    @step
    def sample_datasets(self):
        import openml

        datasets = openml.datasets.list_datasets(output_format="dataframe")
        datasets = datasets[
            (datasets["NumberOfNumericFeatures"] >= 1)
            & (datasets["NumberOfInstances"] <= self.max_obs)
            & (datasets["format"].isin(["Sparse_ARFF", "ARFF"]))
        ]

        # OpenML repository includes many duplicates/extremely similar datasets. try to get rid of some of those here
        datasets = datasets.drop_duplicates(subset="name", keep="first")

        datasets = datasets.sample(
            n=self.num_datasets, random_state=self.random_seed, replace=False
        )
        dataset_ids = datasets["did"].tolist()

        self.datasets = datasets
        self.dataset_ids = dataset_ids
        self.next(self.download_dataset, foreach="dataset_ids")

    @step
    def download_dataset(self):
        import openml

        print("Downloading dataset '{0}'".format(self.input))
        current_dataset = openml.datasets.get_dataset(self.input)
        current_df = current_dataset.get_data(dataset_format="dataframe")[0]

        self.current_df = current_df
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        current_df = self.current_df.select_dtypes(include=numeric_types).reset_index(
            drop=True
        )

        # for huge datasets we take fewer columns to limit simulation runtime
        if len(current_df) < self.n_def_large_datasets:
            current_df = current_df.sample(
                n=min(self.max_cols, len(current_df.columns)),
                axis=1,
                random_state=self.random_seed,
            )
        else:
            current_df = current_df.sample(
                n=min(self.max_cols_large_datasets, len(current_df.columns)),
                axis=1,
                random_state=self.random_seed,
            )

        self.current_df = current_df
        self.next(self.calculate_variances)

    @step
    def calculate_variances(self):
        print(
            "Calculating {} variances for dataset '{}' (number of rows {})".format(
                len(self.current_df.columns), self.input, len(self.current_df)
            )
        )
        import pandas as pd
        import numpy as np

        data_variances = []
        for c in self.current_df.columns:
            data_current = (
                self.current_df[[c]]
                .apply(pd.to_numeric)
                .reset_index(drop=False)
                .rename(columns={"index": "row_id", c: "value"})
            )
            data_current = data_current.dropna().reset_index(drop=True)

            if len(data_current) < 2:
                continue

            # calculate variance using Welford's algorithm
            welford_stats = [(0.0, 0.0, 0.0)]

            for i, r in data_current.iterrows():
                welford_stats.append(update(welford_stats[-1], r["value"]))
            welford_stats = welford_stats[1:]

            data_current = pd.concat(
                [
                    data_current,
                    pd.DataFrame(
                        welford_stats,
                        columns=["count_welford", "mean_welford", "M2_welford"],
                    ),
                ],
                axis=1,
            )

            data_current["var_welford"] = data_current["M2_welford"] / (
                data_current["count_welford"] - 1
            )
            data_current = data_current.assign(column=c)

            # calculate variance using numpy implementation
            data_current["var_numpy"] = np.nan
            data_current["var_numpy"][-1:] = np.var(
                data_current["value"].values, ddof=1
            )

            # only keep last row
            data_current = data_current.tail(1)

            data_variances.append(data_current)

        data_variances = pd.concat(data_variances).assign(dataset_id=self.input)

        self.data_variances = data_variances
        self.next(self.join)

    @step
    def join(self, inputs):
        import pandas as pd

        data_variances_combined = pd.concat([i.data_variances for i in inputs])

        self.merge_artifacts(inputs, exclude=["current_df", "data_variances"])

        print(
            "Calculated a total of {} variances for comparison.".format(
                len(data_variances_combined)
            )
        )

        self.data_variances_combined = data_variances_combined
        self.next(self.end)

    @step
    def end(self):
        print("Simulation finished.")


if __name__ == "__main__":
    WelfordSimFlow()
