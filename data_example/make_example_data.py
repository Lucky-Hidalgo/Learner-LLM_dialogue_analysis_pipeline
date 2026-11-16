"""Generate the SYNTHETIC example dataset in this folder.

Every conversation, identifier and grade produced here is invented. The files exist to show the format
the notebooks expect (see ../README.md) and to let the pipeline be run without the study data.

    python make_example_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)
CODING_SCHEMES = ['lin_dialogue_acts', 'acosta_epistemic_stance', 'del_bosque_cma']
KEYS = ['activity_id', 'user_id', 'role', 'message_num', 'utterance_num']

# Invented sentences: (French original, English translation, [DA label, ES label, CMA label]).
TEMPLATES = {
    'greeting': [("Bonjour ! Je suis ton tuteur, réfléchissons ensemble à ces questions :",
                  "Hello! I am your tutor, let's think about these questions together:",
                  ['Greeting', 'Other acts that maintain discussion',
                   'Tutor Floor Initiating or Floor Mantaining'])],
    'open_question': [("Quelles sont les 2 à 5 idées principales de ce cours ?",
                       "What are the 2 to 5 main ideas of this lesson?",
                       ['Open Question', 'Utterances that are hedges or requests',
                        'Tutor Elicit Response']),
                      ("Quels points te semblent encore obscurs ?",
                       "Which points still seem unclear to you?",
                       ['Open Question', 'Utterances that are hedges or requests',
                        'Tutor Elicit Response'])],
    'directive': [("N'hésite pas à partager ta réflexion !", "Feel free to share your thoughts!",
                   ['Directive', 'Other acts that maintain discussion',
                    'Tutor Floor Initiating or Floor Mantaining'])],
    'user_info': [("Nous avons vu le premier principe et la notion d'énergie interne.",
                   "We saw the first law and the notion of internal energy.",
                   ['Information', 'Utterances that index information',
                    'Student Compound Contribution']),
                  ("Le travail dépend du chemin suivi, contrairement à l'énergie interne.",
                   "Work depends on the path taken, unlike internal energy.",
                   ['Information', 'Utterances that index information',
                    'Student Compound Contribution']),
                  ("Une transformation isotherme garde la température constante.",
                   "An isothermal process keeps the temperature constant.",
                   ['Observation', 'Replies that provide content-based information',
                    'Student Compound Contribution'])],
    'user_question': [("Je ne comprends pas bien la différence entre chaleur et travail.",
                       "I don't really understand the difference between heat and work.",
                       ['Information Question', 'Utterances that are hedges or requests',
                        'Student Question'])],
    'user_short': [("D'accord, merci.", "Okay, thanks.",
                    ['Acknowledge', 'Basic and simple replies to questions', 'Student Backchannel']),
                   ("Oui.", "Yes.",
                    ['Yes-No Answer', 'Basic and simple replies to questions',
                     'Student Backchannel'])],
    'feedback': [("Très bien, c'est un bon résumé !", "Very good, that's a good summary!",
                  ['General Positive Feedback', 'Other acts that maintain discussion',
                   'Acknowledge Backchannel']),
                 ("Ta définition du travail est correcte.", "Your definition of work is correct.",
                  ['Elaborated Positive Feedback', 'Replies that provide content-based information',
                   'Acknowledge Backchannel'])],
    'probing': [("Comment le premier principe s'applique-t-il à un cycle ?",
                 "How does the first law apply to a cycle?",
                 ['Probing Question', 'Utterances that are hedges or requests',
                  'Tutor Diagnosing Base'])],
    'explanation': [("La chaleur et le travail sont deux façons d'échanger de l'énergie.",
                     "Heat and work are two ways of exchanging energy.",
                     ['Explanation', 'Replies that provide content-based information',
                      'Tutor Question Response'])],
}

# Three conversation styles, so that the clustering has some structure to find.
STYLES = {
    'short': [('model', ['greeting', 'open_question', 'open_question', 'directive']),
              ('user', ['user_info', 'user_info']),
              ('model', ['feedback'])],
    'questioning': [('model', ['greeting', 'open_question', 'open_question', 'directive']),
                    ('user', ['user_info', 'user_question']),
                    ('model', ['explanation', 'probing']),
                    ('user', ['user_short']),
                    ('model', ['explanation', 'feedback']),
                    ('user', ['user_short'])],
    'extended': [('model', ['greeting', 'open_question', 'open_question', 'directive']),
                 ('user', ['user_info']),
                 ('model', ['feedback', 'probing']),
                 ('user', ['user_info', 'user_info']),
                 ('model', ['feedback', 'probing']),
                 ('user', ['user_info']),
                 ('model', ['feedback', 'directive'])],
}

for folder in ('source', 'source/human_annotations', 'validation/llm_rerun',
               'validation/unmapped_labels'):
    (HERE / folder).mkdir(parents=True, exist_ok=True)

activities = pd.DataFrame({'activity_id': ['act_week2', 'act_week4', 'act_week5'],
                           'activity_name': ['Summary 2', 'Summary 4', 'Summary 5'],
                           'week': [2, 4, 5],
                           'graded': [False, True, True]})
students = [f'student{i:02d}' for i in range(1, 23)]

sentences, labels = [], {scheme: [] for scheme in CODING_SCHEMES}
for activity in activities['activity_id']:
    for user in students:
        if RNG.random() < 0.25:  # not every student takes part in every activity
            continue
        style = STYLES[RNG.choice(list(STYLES))]
        for message_num, (role, kinds) in enumerate(style):
            for sentence_num, kind in enumerate(kinds):
                french, english, scheme_labels = TEMPLATES[kind][RNG.integers(len(TEMPLATES[kind]))]
                sentences.append({'activity_id': activity, 'user_id': user, 'role': role,
                                  'message_num': message_num, 'sentence_num': sentence_num,
                                  'sentence_orig': french, 'sentence_en': english})
                for scheme, label in zip(CODING_SCHEMES, scheme_labels):
                    labels[scheme].append({'activity_id': activity, 'user_id': user, 'role': role,
                                           'message_num': message_num, 'utterance_num': sentence_num,
                                           'sentence_en': english, 'label': label})

pd.DataFrame(sentences).to_csv(HERE / 'source' / 'conversations.csv', index=False)
activities.to_csv(HERE / 'source' / 'activities.csv', index=False)

consent = pd.DataFrame({'user_id': students + ['student23', 'student24'],
                        'consent_use': 'YES', 'consent_grade': 'YES', 'consent_share': 'YES'})
consent.loc[consent['user_id'] == 'student22', 'consent_share'] = 'NO'  # excluded by notebook 00
consent.to_csv(HERE / 'source' / 'consent.csv', index=False)

pd.DataFrame({'user_id': students,
              'final_grade': RNG.normal(12, 3, len(students)).clip(0, 20).round(1)}) \
    .to_csv(HERE / 'source' / 'grades.csv', index=False)

conversations = pd.DataFrame(sentences)[['activity_id', 'user_id']].drop_duplicates()
graded = conversations[conversations['activity_id'].isin(
    activities.loc[activities['graded'], 'activity_id'])].copy()
graded['summary_given_grade'] = RNG.integers(0, 4, len(graded))
graded['summary_weighted_grade'] = ((graded['summary_given_grade'] + RNG.random(len(graded))) / 4).round(3)
graded.to_csv(HERE / 'source' / 'summary_grades.csv', index=False)

for scheme in CODING_SCHEMES:
    first_run = pd.DataFrame(labels[scheme])

    # Labels exactly as a model would return them, including a lexical variant of a valid label.
    unmapped = first_run.copy()
    if scheme == 'lin_dialogue_acts':
        unmapped.loc[unmapped.sample(frac=0.02, random_state=1).index, 'label'] = 'Acknowledgement'
    unmapped.to_csv(HERE / 'validation' / 'unmapped_labels' / f'{scheme}.csv', index=False)
    unmapped.to_csv(HERE / 'source' / f'llm_labels_{scheme}.csv', index=False)

    # Second run: the same utterances with about 10% of the labels changed.
    rerun = first_run.copy()
    changed = rerun.sample(frac=0.10, random_state=2).index
    rerun.loc[changed, 'label'] = rerun['label'].sample(len(changed), random_state=3).values
    rerun.to_csv(HERE / 'validation' / 'llm_rerun' / f'{scheme}.csv', index=False)

    # Human annotation of a sample of whole conversations, with about 15% disagreement.
    sample = conversations.sample(4, random_state=4)
    human = first_run.merge(sample, on=['activity_id', 'user_id']).rename(
        columns={'label': 'human_label'})
    changed = human.sample(frac=0.15, random_state=5).index
    human.loc[changed, 'human_label'] = human['human_label'].sample(len(changed),
                                                                    random_state=6).values
    human[KEYS + ['sentence_en', 'human_label']].to_csv(
        HERE / 'source' / 'human_annotations' / f'{scheme}.csv', index=False)

print(f'{len(conversations)} synthetic conversations, {len(sentences)} sentences written to {HERE}')
