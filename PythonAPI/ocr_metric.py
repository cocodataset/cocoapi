import re
from difflib import SequenceMatcher

from rapidfuzz import string_metric


def cal_true_positive_char(pred, gt):
    """Calculate correct character number in prediction.

    Args:
        pred (str): Prediction text.
        gt (str): Ground truth text.

    Returns:
        true_positive_char_num (int): The true positive number.
    """

    all_opt = SequenceMatcher(None, pred, gt)
    true_positive_char_num = 0
    for opt, _, _, s2, e2 in all_opt.get_opcodes():
        if opt == 'equal':
            true_positive_char_num += (e2 - s2)
        else:
            pass
    return true_positive_char_num


def count_matches(pred_texts, gt_texts):
    """Count the various match number for metric calculation.

    Args:
        pred_texts (list[str]): Predicted text string.
        gt_texts (list[str]): Ground truth text string.

    Returns:
        match_res: (dict[str: int]): Match number used for
            metric calculation.
    """
    match_res = {
        'gt_char_num': 0,
        'pred_char_num': 0,
        'true_positive_char_num': 0,
        'gt_word_num': 0,
        'match_word_num': 0,
        'match_word_ignore_case': 0,
        'match_word_ignore_case_symbol': 0
    }
    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
    norm_ed_sum = 0.0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        if not pred_text:
            continue
        if gt_text == pred_text:
            match_res['match_word_num'] += 1
        gt_text_lower = gt_text.lower()
        pred_text_lower = pred_text.lower()
        if gt_text_lower == pred_text_lower:
            match_res['match_word_ignore_case'] += 1
        gt_text_lower_ignore = comp.sub('', gt_text_lower)
        pred_text_lower_ignore = comp.sub('', pred_text_lower)
        if gt_text_lower_ignore == pred_text_lower_ignore:
            match_res['match_word_ignore_case_symbol'] += 1
        match_res['gt_word_num'] += 1

        # normalized edit distance
        edit_dist = string_metric.levenshtein(pred_text_lower_ignore,
                                              gt_text_lower_ignore)
        norm_ed = float(edit_dist) / max(1, len(gt_text_lower_ignore),
                                         len(pred_text_lower_ignore))
        # print('Just listing this here, edit_distance:', pred_text, gt_text, norm_ed)
        norm_ed_sum += norm_ed

        # number to calculate char level recall & precision
        match_res['gt_char_num'] += len(gt_text_lower_ignore)
        match_res['pred_char_num'] += len(pred_text_lower_ignore)
        true_positive_char_num = cal_true_positive_char(
            pred_text_lower_ignore, gt_text_lower_ignore)
        match_res['true_positive_char_num'] += true_positive_char_num

    normalized_edit_distance = norm_ed_sum / max(1, len(gt_texts))
    match_res['ned'] = normalized_edit_distance

    return match_res


def eval_ocr_metric(pred_texts, gt_texts):
    """Evaluate the text recognition performance with metric: word accuracy and
    1-N.E.D. See https://rrc.cvc.uab.es/?ch=14&com=tasks for details.

    Args:
        pred_texts (list[str]): Text strings of prediction.
        gt_texts (list[str]): Text strings of ground truth.

    Returns:
        eval_res (dict[str: float]): Metric dict for text recognition, include:
            - word_acc: Accuracy in word level.
            - word_acc_ignore_case: Accuracy in word level, ignore letter case.
            - word_acc_ignore_case_symbol: Accuracy in word level, ignore
                letter case and symbol. (default metric for
                academic evaluation)
            - char_recall: Recall in character level, ignore
                letter case and symbol.
            - char_precision: Precision in character level, ignore
                letter case and symbol.
            - 1-N.E.D: 1 - normalized_edit_distance.
    """
    assert isinstance(pred_texts, list)
    assert isinstance(gt_texts, list)
    assert len(pred_texts) == len(gt_texts)

    match_res = count_matches(pred_texts, gt_texts)
    eps = 1e-8
    char_recall = 1.0 * match_res['true_positive_char_num'] / (
        eps + match_res['gt_char_num'])
    char_precision = 1.0 * match_res['true_positive_char_num'] / (
        eps + match_res['pred_char_num'])
    word_acc = 1.0 * match_res['match_word_num'] / (
        eps + match_res['gt_word_num'])
    word_acc_ignore_case = 1.0 * match_res['match_word_ignore_case'] / (
        eps + match_res['gt_word_num'])
    word_acc_ignore_case_symbol = 1.0 * match_res[
        'match_word_ignore_case_symbol'] / (
            eps + match_res['gt_word_num'])

    eval_res = {}
    eval_res['word_acc'] = word_acc
    eval_res['word_acc_ignore_case'] = word_acc_ignore_case
    eval_res['word_acc_ignore_case_symbol'] = word_acc_ignore_case_symbol
    eval_res['char_recall'] = char_recall
    eval_res['char_precision'] = char_precision
    eval_res['1-N.E.D'] = 1.0 - match_res['ned']

    for key, value in eval_res.items():
        eval_res[key] = float('{:.4f}'.format(value))

    return eval_res
