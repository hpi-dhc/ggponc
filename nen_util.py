import pandas as pd
from tqdm.auto import tqdm

def get_annotation_dataframe(documents, kb, max_cui, threshold, show_progress=True):
    anno_df = []
    for d in tqdm(documents, disable=not show_progress):
        for e in d['entities']:
            if len(e['normalized']) == 0:
                anno_df.append({
                     'document' : d['document_id'],
                     'text' : ' '.join(e['text']),
                     'type' : e['type'],
                     'start' : e['offsets'][0][0],
                     'end' : e['offsets'][-1][1],                  
                    })
            else:
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