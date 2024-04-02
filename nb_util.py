from IPython.display import display, display_markdown, Markdown, HTML
import difflib
from bs4 import BeautifulSoup

def html_diff(a, b):
    sm = difflib.SequenceMatcher(None, a, b)
    result = []

    # Define the CSS styles for different types of changes
    styles = {
        'replace': 'background-color: red; text-decoration: line-through;',
        'delete': 'background-color: red; text-decoration: line-through;',
        'insert': 'background-color: lightgreen;',
        'equal': ''
    }

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        style = styles[tag]
        if tag in ('replace', 'delete'):
            result.append(f'<span style="{style}">{a[i1:i2]}</span>')
        if tag in ('replace', 'insert'):
            result.append(f'<span style="{styles["insert"]}">{b[j1:j2]}</span>')
        elif tag == 'equal':
            result.append(a[i1:i2])

    # Combine the HTML snippets into a full document
    html_result = ''.join(result)
    html_result = f'<html><body><div>{html_result}</div></body></html>'
    display(HTML(html_result))
    
def show_diff(df):
    for i, r in df.reset_index().iterrows():
        display(Markdown(f"{i};{r.file}"))
        display(Markdown(r.sentence))
        display(Markdown(r.sentence_preprocessed))
        html_diff(r.sentence, r.sentence_preprocessed)
        
def show_diff_single(a, b):
    display(Markdown(a))
    display(Markdown(b))
    html_diff(a, b)