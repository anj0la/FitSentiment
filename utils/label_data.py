from constants.constants import WORKOUT_CLASSES, UPPER_BODY_PARTS, LOWER_BODY_PARTS, CORE_PARTS, FULL_BODY_KEYWORDS, UPPER_LOWER_KEYWORDS, PUSH_PULL_LEGS_KEYWORDS

def _label_data(corpus: list[str]) -> list[str]:
        classes = []
        for sentence in corpus:
            words = sentence.split()
            
            # initializing flags for each type of workout              
            has_lower_body = any(word in LOWER_BODY_PARTS for word in words)
            has_upper_body = any(word in UPPER_BODY_PARTS for word in words)
            has_core = any(word in CORE_PARTS for word in words)
            has_full_body = any(word in FULL_BODY_KEYWORDS for word in words) or (has_lower_body and has_upper_body and has_core)
            has_upper_lower = any(word in UPPER_LOWER_KEYWORDS for word in words)
            has_push_pull_legs = any(word in PUSH_PULL_LEGS_KEYWORDS for word in words)
            
            # case one: full body (contains at least one lower and upper body part and core/full body keywords)
            if has_full_body:
                classes.append(WORKOUT_CLASSES[0])  # class 0 = full body
            # case two: upper/lower split
            elif has_upper_lower:
                classes.append(WORKOUT_CLASSES[1])  # class 1 = upper lower split
            # case three: push/pull/legs
            elif has_push_pull_legs: # change
                classes.append(WORKOUT_CLASSES[2])  # class 2 = push pull legs
            # case four: lower body
            elif has_lower_body:
                classes.append(WORKOUT_CLASSES[3])  # class 3 = lower body
            # case five: upper body
            elif has_upper_body:
                classes.append(WORKOUT_CLASSES[4])  # class 4 = upper body
            else:
                classes.append(WORKOUT_CLASSES[5])  # if none of the above cases match, assume it is general fitness
        return classes
    
def create_text_label_rows(corpus: list[str]) -> list[dict]:
    rows = []
    labels = _label_data(corpus)
    for sentence, label in zip(corpus, labels):
        rows.append({'text': sentence, 'label': label})
    return rows
