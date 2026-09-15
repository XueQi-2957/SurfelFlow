# GFlow

GFlow recovers a 4D world from monocular video. This repository contains the
core research implementation and third-party submodule configuration.

## Installation

```bash
git clone --recursive https://github.com/XueQi-2957/SurfelFlow.git
cd SurfelFlow
pip install -r requirements.txt
```

External components listed in `.gitmodules`, model checkpoints, and datasets
must be installed separately according to their upstream instructions. They
are intentionally excluded from this repository.

## Core entry points

- `gflow/fit_video.py`: streaming video fitting and optimization.
- `gflow/trainer.py`: training logic and Gaussian/surface state management.
- `gflow/benchmark.py`: evaluation for one sequence.
- `gflow/benchmark_multi.py`: evaluation across a dataset.
- `gflow/viewer.py`: local result viewer.
- `gflow/geometry_eval.py`: geometry metric utilities.

Example commands:

```bash
python gflow/fit_video.py --sequence_path /path/to/sequence
python gflow/viewer.py --folder /path/to/log_dir --gpu 0 --port 8088
```

Experimental launchers, tests, internal plans, datasets, generated results,
manuscript material, and local build environments are maintained outside this
public release.

## Citation

If you use GFlow, please cite the associated AAAI 2025 paper:

```bibtex
@article{gflow2024,
  title={GFlow: Recovering 4D World from Monocular Video},
  author={Wang, Shizun and Yang, Xingyi and Shen, Qiuhong and Jiang, Zhenxiang and Wang, Xinchao Wang},
  year={2025}
}
```
