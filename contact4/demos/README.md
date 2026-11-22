# Demonstrations

The demonstrations are not stored in this repository. Download them from

```
https://huggingface.co/datasets/ntumars/Contact-4
```

and place its content here. With the Hugging Face command line tool, run in this folder

```
huggingface-cli download ntumars/Contact-4 --repo-type dataset --local-dir .
```

```
demos/
├── peg_insertion/     Insert, 50 episodes, 4972 frames
├── wipe_table/        Wipe, 50 episodes, 14018 frames
├── write_board/       Write, 50 episodes, 47814 frames
├── pivot_box/         Pivot, 50 episodes, 6429 frames
└── reference/
    ├── place_bowl/    Place, 50 episodes, 10704 frames
    └── stack_cubes/   Stack, 50 episodes, 9258 frames
```

The four contact-rich tasks are at the top level, and the two pick-and-place reference tasks are in `reference/`. Each task folder is one LeRobot v2.1 dataset with `data/chunk-000/episode_*.parquet` and `meta/`. The download contains a README with the full description of the format.
