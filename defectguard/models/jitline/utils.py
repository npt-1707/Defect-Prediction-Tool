import re
import pickle
from defectguard.utils.utils import SRC_PATH

with open(f'{SRC_PATH}/models/jitline/common_tokens.pkl', 'rb') as f:
    common_tokens = pickle.load(f)

def preprocess_code_line(code, remove_common_tokens = True, language='python'):
    assert language in common_tokens, f"JITLine: Language not supported. Only support: {', '.join(common_tokens.keys())}"
    code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']', ' ').replace(
        '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')
    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)

    # remove continuous whitespace
    code = code.split()
    code = ' '.join(code)
    if remove_common_tokens:
        new_code = ' '.join([tok for tok in code.split() if tok not in common_tokens[language]])
        return new_code.strip()
    else:
        return code.strip()


def preprocess_code_diff(code_diff, remove_common_tokens = True, language='python'):
    combined_code = []
    for commit_code in code_diff:
        hunk_added = []
        hunk_removed = []
        for hunk in commit_code:
            hunk_added.extend([preprocess_code_line(line, remove_common_tokens, language) for line in hunk["added_code"]])
            hunk_removed.extend([preprocess_code_line(line, remove_common_tokens, language) for line in hunk["removed_code"]])
        hunk_added = " \n ".join(list(set(hunk_added)))
        hunk_removed = " \n ".join(list(set(hunk_removed)))
        combined_code.append(hunk_added + " " + hunk_removed)
    return combined_code