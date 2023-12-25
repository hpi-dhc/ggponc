import typer
from pathlib import Path
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy import util
from thinc.api import Config
import wandb
import chunk_and_ngram_suggester

import spacy
spacy.require_gpu(0)

def main(default_config: Path, output_path: Path):
    loaded_local_config = util.load_config(default_config)
    with wandb.init() as run:
        sweeps_config = Config(util.dot_to_dict(run.config))
        merged_config = Config(loaded_local_config).merge(sweeps_config)
        nlp = init_nlp(merged_config)
        output_path = output_path / run.id
        print(f"Saving to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        train(nlp, output_path, use_gpu=True)


if __name__ == "__main__":
    typer.run(main)