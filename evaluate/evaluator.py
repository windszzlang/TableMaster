import re
import unicodedata


# references
# https://github.com/Leolty/tablellm/blob/aef85050f522900fd70920c2b7427a383e3066ab/utils/eval.py
# https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py



def stringify(x):
    if x is None:
        x = ''
    if not isinstance(x, str):
        x = str(x)
    return x


def normalize_number(x: str):
    def convert_match(match):
        num = match.group(0)
        if num != '.':
            return str(float(num))
        return num

    pattern = r'[+-]?[0-9]*[.]?[0-9]+' # any number
    return re.sub(pattern, convert_match, x)


def normalize(x):
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x) if unicodedata.category(c) != 'Mn')
    # Remove return
    x = re.sub(r'\n', ' ', x)
    # Remove star
    x = re.sub(r'\*', '', x)
    # Normalize or remove quotes and dashes
    x = re.sub(r"[‘’´`']", "", x)
    x = re.sub(r"[“”\"]", "", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    # Remove dash unless it is a negative sign
    x = re.sub(r'-(?!\d)', ' ', x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Convert to lowercase
    x = x.lower()
    # Remove commas between digits
    x = re.sub(r',', '', x)
    # Remove everything before and including "answer:"
    while True:
        if 'answer:' in x:
            x = re.sub(r'^.*?answer:', '', x)
        else:
            break
    # Remove articles
    x = re.sub(r'\b(a|an|the)\b', ' ', x)
    # Remove unit
    x = re.sub(r'^([+-]?[0-9]*[.]?[0-9]+) \w+$', r'\1', x)
    # Normalize number
    x = normalize_number(x)
    # Collapse whitespaces
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def normalize_list(x):
    '''
    gt_answer in a list: a|b|c|d
    '''
    x = re.sub(r'\|', ' ', x)
    x = re.sub(r', ', ' ', x)
    x = re.sub(r' and ', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x


def normalize_fin_number(x):
    x = stringify(x)
    x = x.lower()
    if "yes" in x or "true" in x:
        return 1
    elif "no" in x or "false" in x:
        return 0
    cleaned_text = re.sub(r'[^\d.-]', '', x).strip()
    try:
        number = float(cleaned_text)
    except:
        return 0
    return number


def remove_parentheses_content(text):
    cleaned_text = re.sub(r'\(.*?\)', '', text)
    return cleaned_text


## Evaluation for Simple Table-based QA
# exact match with answer normalization in table qa
def eval_qa(pred_answer: str, gt_answer: str):
    '''
    pred_answer: str
    gt_answer: str
    '''
    pred_answer = stringify(pred_answer)
    gt_answer = stringify(gt_answer)
    if '|' in gt_answer:
        pred_answer = normalize_list(pred_answer)
        gt_answer = normalize_list(gt_answer)

    normalized_pred_answer = normalize(pred_answer)
    normalized_gt_answer = normalize(gt_answer)
    res = normalized_pred_answer == normalized_gt_answer
    return res



## Evaluation for Table-based Fact Verfication
def eval_fact(pred_answer: str, gt_answer: bool):
    '''
    pred_answer: str
    gt_answer: bool
    '''
    pred_answer = str(pred_answer).lower()
    if 'no' in pred_answer or 'false' in pred_answer:
        normalized_pred_answer = False
    elif 'yes' in pred_answer or 'true' in pred_answer:
        normalized_pred_answer = True
    else:
        normalized_pred_answer = False
    res = normalized_pred_answer == gt_answer
    return res



## Evaluation for Table-based Free-form QA
def eval_free_qa(predicted_answer_list, answer_list):
    """
    Evaluate predictions using SacreBLEU and ROUGE-1, ROUGE-2, and ROUGE-L.

    Args:
        predicted_answer_list (list of str): List of predicted texts.
        answer_list (list of str): List of reference texts.

    Returns:
        dict: Dictionary containing SacreBLEU, ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    assert len(predicted_answer_list) == len(answer_list), "Predicted answers and answers must have the same length."

    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import sacrebleu
    from rouge_score import rouge_scorer
    nltk.download('punkt', quiet=True)

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Metrics containers
    sacrebleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Iterate over predictions and references
    for i, (ref, pred) in enumerate(zip(answer_list, predicted_answer_list)):
        # SacreBLEU score with specified configuration
        ref = normalize(ref)
        pred = normalize(pred)
        ref = remove_parentheses_content(ref)
        pred = remove_parentheses_content(pred)

        sacrebleu_score = sacrebleu.sentence_bleu(
            pred,
            [ref],
            smooth_method="exp",
            tokenize="13a",
            lowercase=True,
            use_effective_order=True
        ).score / 100
        sacrebleu_scores.append(sacrebleu_score)
        # ROUGE scores
        rouge_scores = rouge.score(pred, ref)
        rouge1_scores.append(rouge_scores["rouge1"].fmeasure)
        rouge2_scores.append(rouge_scores["rouge2"].fmeasure)
        rougeL_scores.append(rouge_scores["rougeL"].fmeasure)


    # Compute average scores
    results = {
        "BLEU": sum(sacrebleu_scores) / len(sacrebleu_scores),
        "ROUGE-1": sum(rouge1_scores) / len(rouge1_scores),
        "ROUGE-2": sum(rouge2_scores) / len(rouge2_scores),
        "ROUGE-L": sum(rougeL_scores) / len(rougeL_scores),
    }

    return results


if __name__ == '__main__':
    print(eval_qa('yes', 'yes'))
    print(eval_fact('True', True))

    x = '2131 (23323)'
    print(x)
    print(normalize(x))