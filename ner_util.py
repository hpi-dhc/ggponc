from pathlib import Path
from spacy.tokens import Doc, DocBin
from datasets import Dataset, DatasetDict
from typing import List, Union
from tqdm.auto import tqdm

def bigbio_split_passages(bigbio_dataset : Dataset) -> Dataset:
    def split_passages(batch):
        result = { "id": [], "document_id" : [], "passages": [], "entities": [], "relations": [], "events": [], "coreferences": [] }
        for di, document_id in enumerate(batch["document_id"]):
            doc_ents = batch["entities"][di]
            for i, p in enumerate(batch["passages"][di]):
                for j, (text, off) in enumerate(zip(p['text'], p['offsets'])):
                    result["document_id"].append(f"{document_id}-{i}-{j}")
                    result["id"].append(f"{batch['id'][di]}-{i}-{j}")

                    # Fix entites and passage offsets
                    passage_offset = [[0, off[1] - off[0]]]

                    result["passages"].append([{"text": [text], "offsets": passage_offset}])
                    result["entities"].append([])

                    remaining_ents = []
                    for e in doc_ents:
                        if e['offsets'][0][0] >= off[0] and e['offsets'][0][1] <= off[1]:
                            e = e.copy()
                            e['offsets'][0][0] -= off[0]
                            e['offsets'][0][1] -= off[0]
                            result["entities"][-1].append(e)
                        else:
                            remaining_ents.append(e)
                    doc_ents = remaining_ents
                    

                    result["relations"].append([])
                    result["events"].append([])
                    result["coreferences"].append([])
        return result
    
    return bigbio_dataset.map(split_passages, batched=True)

def bigbio_to_spacy(nlp, bigbio_dataset : Dataset, span_key : str, is_sentencized : bool = False) -> List[Doc]:
    result = []
    def join_passages(d):
        return " ".join([t for p in d["passages"] for t in p["text"]])
    doc_texts = [join_passages(d) for d in bigbio_dataset]
    for doc, d in zip(nlp.pipe(doc_texts), tqdm(bigbio_dataset)):        
        spans = []
        for e in d['entities']:            
            span = doc.char_span(e['offsets'][0][0], e['offsets'][-1][1], label=e['type'], alignment_mode='expand')
            assert span
            if span:
                spans.append(span)
        doc.spans[span_key] = spans
        doc.user_data['document_id'] = d['document_id']
        if is_sentencized:
            set_sentence_boundaries(doc)
        result.append(doc)
    return result

def bigbio_to_spacy_docbin(folder : Union[str, Path], nlp, bigbio_dataset : DatasetDict, span_key : str, is_sentencized : bool = False):
    out = Path(folder)
    out.mkdir(exist_ok=True)
    for split, dataset in bigbio_dataset.items():
        docs = bigbio_to_spacy(nlp, dataset, span_key, is_sentencized)
        doc_bin = DocBin(docs=docs, store_user_data=True)
        doc_bin.to_disk(out / f"{split}.spacy")


def set_sentence_boundaries(doc):
    doc[0].is_sent_start = True
    for t in doc[1:]:
        t.is_sent_start = False