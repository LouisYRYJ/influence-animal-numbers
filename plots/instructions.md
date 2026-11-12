# Plotting filtering results

The plotting scripts read the evaluation CSVs produced by
`scripts/run_filtering_experiment_eval_sliced.sh`. They do not assume a
repository location: put paths in a JSON config, relative to that config file
or as absolute paths.

## Panel configuration

Create `plot_config.json` with one panel per subplot:

```json
{
	"panels": [
		{
			"title": "Dog - sample filtering",
			"animal": "dog",
			"eval_directory": "filtering_results_sliced_ekfac/attribution/ONE_WORD/0/unsloth/Qwen2.5-3B-Instruct/dog/evals",
			"base_model_csv": "base_results/unsloth/Qwen2.5-3B-Instruct/base_model.csv",
			"full_data_csv": "base_results/unsloth/Qwen2.5-3B-Instruct/dog/evals/dog_teacher_numbers.csv",
			"num_seeds": 10
		}
	]
}
```

For `plot_2x2_grid.py` and the compact sliced variant, provide exactly four
panels. The `eval_directory` may point below either
`filtering_results_sliced*` / `filtering_results_tok_sliced*` or
`filtering_results_cumulative*` / `filtering_results_tok_cumulative*`.
For cumulative plots, the eval directory must contain files named
`top_indices_<fraction>_<seed>.csv` and `bottom_indices_<fraction>_<seed>.csv`.
For decile plots, use the `slice_XX_pct_YY_ZZ_seed.csv` naming produced by the
filtering scripts. Anchor CSVs are optional; omit them if no base/full model
evaluation is available.

## Commands

Run commands from any directory. The output path is required and parent
directories are created automatically:

```bash
python plots/plot_2x2_grid.py plot_config.json --output figures/sliced.png
python plots/plot_2x2_cumulative.py plot_config.json --output figures/cumulative.pdf
python plots/plot_2x2_grid_compact.py plot_config.json --output figures/sliced_compact.png
python plots/plot_2x2_cumulative_compact.py plot_config.json --output figures/cumulative_compact.png
```

Use `--show-random` for sliced plots or omit `--no-random` for cumulative
plots when random-baseline files are present. The single-panel and 3-row
variants use the same schema; the single-panel config has one panel, while the
3-row config has ordered sample/token pairs.

## Bar charts

Bar charts use a separate config because they compare several models and
categories. Each model entry supplies its result-tree model directory and the
animals to plot:

```json
{
	"results_root": ".",
	"filtering_mode": "sliced",
	"base_results": "base_results",
	"models": {
		"Qwen": {
			"directory": "unsloth/Qwen2.5-3B-Instruct",
			"concepts": ["cat", "dog"]
		}
	}
}
```

The result root must contain the standard sliced result directories. Set
`filtering_mode` to `cumulative` only to make the directory naming explicit;
bar charts still require sliced CSVs because they compare the first and last
deciles. Use the cumulative panel plotters for
`filtering_results_cumulative*` and `filtering_results_tok_cumulative*`.
Run either:

```bash
python plots/bar_chart.py bar_config.json --output figures/bars.png --normalise
python plots/bar_chart_per_animal.py bar_config.json --output figures/bars_by_animal.png
```

Missing CSVs are reported as empty data for that bar; verify the result paths
and seed count when a category is unexpectedly empty.
