import spacy
from spacy.pipeline import spancat
from spacy.pipeline.spancat import Suggester

from spacy.training import Example
from typing import Optional, Iterable, Dict, Set, List, Any, Callable, Tuple, cast
from thinc.api import Optimizer, Ops, get_current_ops
from thinc.types import Ragged, Ints2d, Floats2d, Ints1d
from spacy.tokens import Doc

def flat_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    """ 
    A scorer that unwraps the nested dict returned by the default spancat scorer, so we can display it during training. 
    """
    scores = spancat.spancat_score(examples=examples, **kwargs)
    res = {}
    for k,v in scores.items():
        if type(v) is dict:
            for k1, v1 in v.items():
                if type(v1) is dict:
                    for k2, v2 in v1.items():
                        res[k1 + '_' + k2] = v2
                else:
                    res[k1] = v1
        else:
            res[k] = v
    return res

@spacy.registry.scorers("phlobo.flat_scorer")
def make_spancat_flatscorer():
    return flat_score

@spacy.registry.misc("phlobo.chunk_and_ngram_suggester")
def build_chunk_and_ngram_suggester(sizes: List[int], max_depth: int) -> Suggester:
    """A suggester that extends the basic n-gram suggester by adding noun chunks.
    
    :param sizes: for item i in the list, add all the token i-grams in each token
    :param max_depth: follow the dependency graph for this many steps. 
    
    If max_depth = 0, only basic noun chunks are considered. 
    If max_depth > 0, the dependency graph is followed recursively for this many steps to connect noun_chunks through their head.
    If max_depth < 0, no noun chunks are considered and only n-grams returned
    
    """
    def chunk_and_ngram_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
        max_sizes = max(sizes)
        
        if ops is None:
            ops = get_current_ops()
        spans = []
        lengths = []
        for i, doc in enumerate(docs):            
            length = 0
            if doc.has_annotation("DEP") and max_depth >= 0:
                chunks = set()
                nc = list(doc.noun_chunks)
                
                for chunk in nc:
                    if (chunk.end - chunk.start) > max_sizes:
                        chunks.add((chunk.start, chunk.end))                            

                    def extend_chunk(head, the_chunk): 
                        for other_chunk in nc:
                            if head in other_chunk:
                                start = min(the_chunk.start, other_chunk.start)
                                end = max(the_chunk.end, other_chunk.end)
                                return other_chunk, doc[start:end]
                        return None, None

                    depth = 0

                    def add_extensions(extension_chunk, cur_depth):        
                        extend1, chunk1 = extend_chunk(extension_chunk.root.head, chunk)

                        if chunk1 and (((chunk1.end - chunk1.start) > max_sizes)):
                            chunks.add((chunk1.start, chunk1.end))
                            if cur_depth < max_depth:
                                add_extensions(extend1, cur_depth + 1)

                        extend2, chunk2 = extend_chunk(extension_chunk.root.head.head, chunk)

                        if chunk2 and (((chunk2.end - chunk2.start) > max_sizes)):
                            chunks.add((chunk2.start, chunk2.end))
                            if cur_depth < max_depth:
                                add_extensions(extend2, cur_depth + 1)

                    add_extensions(chunk, 0)

                chunks = [list(c) for c in chunks]
                    
                for c in chunks.copy():
                    if (c[1] - c[0]) > 1:
                        chunks.append([c[0] + 1, c[1]])      
                            
                chunks = ops.asarray(chunks, dtype="i")
                
                if chunks.shape[0] > 0:
                    spans.append(chunks)
                    length += chunks.shape[0]
            
            # Add n-grams
            starts = ops.xp.arange(len(doc), dtype="i")
            starts = starts.reshape((-1, 1))
            for size in sizes:
                if size <= len(doc):
                    starts_size = starts[: len(doc) - (size - 1)]
                    spans.append(ops.xp.hstack((starts_size, starts_size + size)))
                    length += spans[-1].shape[0]
                if spans:
                    assert spans[-1].ndim == 2, spans[-1].shape            
            
            lengths.append(length)
                    
        lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
        if len(spans) > 0:
            output = Ragged(ops.xp.vstack(spans), lengths_array)
        else:
            output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)
 
        assert output.dataXd.ndim == 2
    
        return output
 
    return chunk_and_ngram_suggester
