import pandas as pd
from tqdm.auto import tqdm

def get_annotation_dataframe(candidates, kb, max_cui, threshold):
    anno_df = []
    for d in tqdm(candidates):
        for e in d['entities']:
            for ni, n in enumerate(e['normalized'][0:max_cui]):
                if ni == 0 or n['score'] > threshold:
                    concept = kb.cui_to_entity[n['db_id']]
                    anno_df.append({
                         'document' : d['document_id'],
                         'text' : ' '.join(e['text']),
                         'type' : e['type'],
                         'start' : e['offsets'][0][0],
                         'end' : e['offsets'][-1][1],
                         'cui' : n['db_id'],
                         'tuis' : concept.types,
                         'canonical' : concept.canonical_name,
                         'linker' : n['predicted_by'],
                         'confidence' : n['score']                     
                     })
    return pd.DataFrame(anno_df)