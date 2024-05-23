from pathlib import Path
from typing import List, Tuple, Any, Union
from itertools import groupby

from tqdm.auto import tqdm
import pandas as pd

import datasets
from transformers import Text2TextGenerationPipeline, AutoModelForSeq2SeqLM, AutoTokenizer, utils
from transformers.pipelines.pt_utils import KeyDataset
from spacy import Language
import spacy
from spacy.tokens import Doc, DocBin

from xmen import load_kb
from xmen.log import logger
from xmen.data import AbbreviationExpander, SemanticTypeFilter, filter_and_apply_threshold
from xmen.linkers import default_ensemble
from xmen.reranking import CrossEncoderReranker

def read_sentences(version, folder='output'):
    res = []
    for f in tqdm(list(sorted(Path(f'{folder}/{version}/plain_text/sentences/all_files_sentences/').glob('*.txt')))):
        with open(f, 'r', encoding='utf-8') as fh:
            for si, l in enumerate(fh.readlines()):
                l = l.rstrip()
                if l:
                    res.append({'file' : f.stem, 'sentence_id': si, 'sentence' : l})
    return pd.DataFrame(res)

def merge_sentence_docs(sentence_docs : List[Doc], group_key : List[Any], key_name='file_name'):
    docs = []
    for key, grp in groupby(zip(sentence_docs, group_key), key=lambda t: t[1]):
        sents = [g[0] for g in grp]
        for s in sents:
            s.user_data = {}
        doc = Doc.from_docs(sents)
        for k in doc.spans.keys():
            assert sum([len(d.spans[k]) for d in sents]) == len(doc.spans[k])
        doc.user_data[key_name] = key
        docs.append(doc)
    return docs

class ECCNPResolver():
    
    def __init__(self, model_path, generation_max_length = 300, batch_size = 16, tokenizer = 'google/mt5-base'):        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.pipeline = Text2TextGenerationPipeline(model, tokenizer=tokenizer, max_length=generation_max_length, device=0)
        self.batch_size = batch_size

    def transform(self, inputs: Union[str,List[str]], show_progress=True):
        is_str = False
        if isinstance(inputs, str):
            is_str = True
            inputs = [ inputs ]
        res = []
        ds = KeyDataset([{'text' : sent} for sent in inputs], 'text')
        for s in tqdm(self.pipeline(ds, batch_size=self.batch_size), total = len(inputs), disable=not show_progress or is_str):
            res.append(s[0]['generated_text'])
        if is_str:
            return res[0]
        return res
        
    def set_df(self, sent_df):
        _sent_df = sent_df.copy()
        _sent_df['sentence_preprocessed'] = self.transform(sent_df.sentence.tolist())
        return _sent_df


@Language.component('prevent-sbd')
def prevent_sentence_boundary_detection(doc):
    doc[0].is_sent_start = True
    for token in doc[1:]:
        token.is_sent_start = False
    return doc

class NERTagger():

    def __init__(self, spacy_model = 'de_ggponc_medbertde', cuda=True, batch_size=256):
        self.batch_size = batch_size
        if cuda:
            spacy.require_gpu()
        self.nlp = spacy.load(spacy_model)
        # GGPONC is already split into sentences
        self.nlp.add_pipe('prevent-sbd', before='parser')

    def transform(self, inputs : Union[str,List[str]], as_tuples = False, show_progress=True):
        is_str = False
        if isinstance(inputs, str):
            is_str = True
            inputs = [ inputs ]
        res = list(self.nlp.pipe(tqdm(inputs, disable=not show_progress or is_str), as_tuples=as_tuples, batch_size=self.batch_size))
        if is_str:
            return res[0]
        return res
    
    def set_df(self, sent_df, sent_col = 'sentence_preprocessed', doc_key = 'file_name' ):
        _sent_df = sent_df.copy()
        _sent_df['spacy_ner'] = self.transform(sent_df[sent_col])
        for _, r in _sent_df.iterrows():
            r.spacy_ner.user_data[doc_key] = f"{r.file}_{r.sentence_id}"
        return _sent_df

class EntityLinker():

    ALL_STEPS =  ['abbrv', 'candidates', 'semantic_type', 'reranking']

    def __init__(self, kb_path, candidate_generation_kwargs = {}, expand_abbrevations = True, k_reranking = 16, st_mapping_file = 'ggponc2tui.csv', reranker = 'phlobo/xmen-de-ce-medmentions', use_nil = False):
        self.kb = load_kb(kb_path)
        self.expand_abbrevations = expand_abbrevations
        self.k_reranking = k_reranking
        self.candidate_generator = default_ensemble(**candidate_generation_kwargs)
        self.use_nil = use_nil
        if st_mapping_file:
            tui_df = pd.read_csv(st_mapping_file)
            type2tui = {}
            for c in ['Diagnosis_or_Pathology', 'Other_Finding', 'Clinical_Drug', 'Nutrient_or_Body_Substance',
                'External_Substance', 'Therapeutic', 'Diagnostic']:
                type2tui[c] = list(tui_df.TUI[tui_df[c] == 'x'].values)
            self.type_filter = SemanticTypeFilter(type2tui, self.kb)
        if reranker:
            self.rr = CrossEncoderReranker.load("phlobo/xmen-de-ce-medmentions", device=0)            
  
    def transform(self, ds, steps = 'all', silent=False) -> dict:
        progress = utils.logging.is_progress_bar_enabled()
        try:
            if silent:
                utils.logging.disable_progress_bar()
            result = datasets.DatasetDict()
            result['dataset'] = ds
            if steps == 'all':
                steps = self.ALL_STEPS
            if 'abbrv' in steps:
                if not silent:            
                    logger.info("Expanding Abbreviations")
                ds = AbbreviationExpander().transform_batch(ds)
            if 'candidates' in steps:
                if not silent:            
                    logger.info("Generating Candidates")
                result['candidates'] = self.candidate_generator.predict_batch(ds, batch_size=128)
            else:
                return result
            if 'semantic_type' in steps:
                if not silent:            
                    logger.info("Filtering Semantic Types")
                result['candidates'] = self.type_filter.transform_batch(result['candidates'])
            if 'reranking' in steps:
                if not silent:            
                    logger.info("Re-ranking Candidates")
                candidates = filter_and_apply_threshold(result['candidates'], self.k_reranking, 0.0)
                ce_candidates = CrossEncoderReranker.prepare_data(candidates, None, self.kb, use_nil=self.use_nil)
                result['reranked'] = self.rr.rerank_batch(candidates, ce_candidates)
            return result
        finally:
            if progress:
                utils.logging.enable_progress_bar()
            