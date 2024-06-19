import argparse
from ggponc import read_sentences, ECCNPResolver, NERTagger, merge_sentence_docs, EntityLinker
import logging
from pathlib import Path
import shutil

from xmen import load_config
from xmen.data import from_spacy

from nen_util import get_annotation_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run the NER and NEN pipeline')
    parser.add_argument('input', type=str, help='Input folder (GGPONC release)')
    parser.add_argument('output', type=str, help='Output folder', nargs='?', default='output')
    parser.add_argument('--unzipped', action='store_true', help='Is the plain text folder unzipped?', default=False)
   
    args = parser.parse_args()
    conf = load_config('xmen_ggponc3.yaml')
    output = Path(args.output)
    version = Path(args.input).name
    output_path = output / version
    logger.info("Using output folder %s", output_path)

    if output_path.exists():
        logger.warning(f"Output folder {output_path} already exists, delete it? (y/n)")
        if input().lower() == 'y':
            shutil.rmtree(output_path)
        else:
            logger.info("Exiting.")
            return 1            
    
    output_path.mkdir(parents=True, exist_ok=True)
  
    sent_df = read_sentences(args.input, zipped=not args.unzipped)
    logger.info("Read %d sentences", len(sent_df))

    logger.info("Resolving ECCNPs")
    resolver = ECCNPResolver(**conf.eccnp)
    sent_df = resolver.set_df(sent_df)

    sent_df.to_parquet(output_path / 'sentences_resolved.parquet')

    # Drop excessively long pre-processed sentences, most like generation errors
    sent_df['ratio'] = (sent_df.sentence_preprocessed.str.len() / sent_df.sentence.str.len()).sort_values()
    drop_index = sent_df.ratio > 2
    logger.info('Resetting', drop_index.sum(), '/', len(drop_index), 'docs due to likely generation errors')
    sent_df.loc[sent_df.ratio > 2, 'sentence_preprocessed'] = sent_df.sentence

    logger.info("Running NER")
    ner = NERTagger()
    ner_df = ner.set_df(sent_df)

    docs = merge_sentence_docs(ner_df.spacy_ner, ner_df.file)
    ds = from_spacy(docs, span_key='entities', doc_id_key='file_name')  

    ds.save_to_disk(str(output_path / 'ggponc_ner_spacy'))

    logger.info(f"Identified entities: {len([e for d in ds for e in d['entities']])}")

    logger.info("Running Entity Linker")
    linker = EntityLinker(**conf.linker.ranking, candidate_generation_kwargs=conf.linker.candidate_generation)
    result = linker.transform(ds)
    result.save_to_disk(str(output_path / 'ggponc_xmen'))

    df = get_annotation_dataframe(result['reranked'], linker.kb, 1, 0.0)
    out_p = output_path / 'predictions'
    out_p.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p / 'silver_standard_entities_linked_xmen.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()