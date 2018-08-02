import pandas as pd
import numpy as np
from math import log



def edit_distance(s1, s2):
    m = len(s1) + 1
    n = len(s2) + 1

    tbl = {}
    for i in range(m): tbl[i,0] = i
    for j in range(n): tbl[0,j] = j

    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i,j-1] + 1,
                           tbl[i-1,j] + 1,
                           tbl[i-1,j-1] + cost)

    return tbl[i,j]


def tfidf_distance(reference):
    tokens = list(x for y in reference for x in y.split())

    tf = np.zeros((len(tokens), len(reference)))
    for i in range(tf.shape[0]):
        for j in range(tf.shape[1]):
            tf[i,j] = reference[j].count(tokens[i]) / len(reference[j].split())

    idf = np.zeros(len(tokens))
    for i in range(len(idf)):
        occurrences = sum(x.find(tokens[i]) != -1 for x in reference)
        idf[i] = log(len(reference) / occurrences, 2)

    tfidf = np.zeros((len(tokens), len(reference)))
    for i in range(tf.shape[0]):
        for j in range(tf.shape[1]):
            tfidf[i,j] = tf[i,j] * idf[i]

    # print(pd.DataFrame(tfidf, index=tokens).round(2))
    return tfidf


def compute_edit_distance_matrix(list1, list2):
    edit_distance_matrix = np.zeros((len(list1), len(list2)))
    for i in range(len(list1)):
        for j in range(len(list2)):
            edit_distance_matrix[i,j] = edit_distance(list1[i], list2[j])
    # print(pd.DataFrame(edit_distance_matrix, index=list1, columns=list2).round(2))
    return edit_distance_matrix


def compute_combined_distance(reference, target):
    reference = [x.lower() for x in reference]
    target = [x.lower() for x in target]

    tokens_reference = list(x for y in reference for x in y.split())
    tokens_target = list(x for y in target for x in y.split())

    tfidf = tfidf_distance(reference)

    ed_matrix = compute_edit_distance_matrix(tokens_reference, tokens_target)

    combined_matrix = np.zeros((len(reference), len(target)))
    for i in range(len(reference)):
        curr_ref = reference[i]
        for j in range(len(target)):
            combined_distance = 0
            curr_tar = target[j]
            token_curr_tar = curr_tar.split()
            token_curr_ref = curr_ref.split()

            for k in range(len(token_curr_tar)):
                try:
                    doc_token = token_curr_tar[k]
                    ref_token = min(token_curr_ref,
                                    key=lambda x: edit_distance(doc_token, x))
                    token_curr_ref.remove(ref_token)

                    ed = ed_matrix[tokens_reference.index(ref_token),
                                         tokens_target.index(doc_token)]

                    tfidf_w = tfidf[tokens_reference.index(ref_token),i]

                    combined_distance += ed * tfidf_w
                except:
                    pass

            combined_matrix[i,j] = combined_distance

    return combined_matrix


reference = [
    "IBM Corporation",
    "AT&T Corporation",
    "Microsoft Corporation",
    "Google Inc",
    "Repubblica Democratica del Congo",
    "Repubblica Democratica di Corea",
    "Repubblica Democratica Tedesca",
    "Associazione Calcio Milan",
    "Torino Football Club",
    "Football Club Internazionale Milano",
]

target = [
    "kongo",
    "korea",
    "milna",
    "intrnazionale",
    "torino",
    "repubblica tedesca",
    "atet",
    "ibm corporatn",
    "microft crpoation",
    "googe",
    "Inc Google",
]

pd.set_option('expand_frame_repr', False)

result = compute_combined_distance(reference, target)
df = pd.DataFrame(result, columns=target, index=reference)

print(df.round(2))
