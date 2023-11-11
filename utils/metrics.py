import nltk
from rouge_score import rouge_scorer


def levenshteinDistance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]


if __name__ == "__main__":
    original = "\\alpha N _ { f } = \\alpha ( \\sum _ { I = 1 } ^ { 9 } v _ { I } \\gamma _ { I } + \\sum _ { i = 1 } ^ { 3 } \\frac { i \\mu x ^ { i } } { 4 } \\{ \\gamma _ { i } , \\gamma _ { 1 2 3 } \\} ) + \\alpha ^ { 2 } \\mu ^ { 2 } / 4 ^ { 2 }"
    # machine_translated = "\\alpha N _ { f } = \\alpha ( \\sum _ { I = 1 } ^ { 9 } v _ { I } \\gamma _ { I } + \\sum _ { i = 1 } ^ { 3 } \\frac { i \\mu x ^ { i } } { 4 } \\{ \\gamma _ { i } , \\gamma _ { 1 2 3 } \\} ) + \\alpha ^ { 2 } \\mu ^ { 2 } / 4 ^ { 2 }"
    machine_translated = "\\alpha N _ { f } = \\alpha ( \\sum _ { l = 1 } ^ { 9 } v _ { I } \\gamma _ { J } + \\sum _ { i = 1 } ^ { 3 } \\frac { i \\mu x ^ { i } } { 4 } \\{ \\gamma _ { i } , \\gamma _ { l 2 3 } \\} ) + \\alpha ^ { 2 } \\mu ^ { 2 } / 4 ^ { 2 }"

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge4'], use_stemmer=True)
    print(nltk.translate.bleu(references=[original.split()], hypothesis=machine_translated.split()))
    print(levenshteinDistance(original, machine_translated))
    score = scorer.score(original, machine_translated)["rouge1"][-1]
    print(scorer.score(original, machine_translated))


#-(8(4a+5b+8c)^2)/(a^2b)-(5(4a+5b+8c)^2)/(a^2c)+(8(4a+5b+8c)^2)/(abc)=0 and -(8(4a+5b+8c)^2)/(ab^2)-(4(4a+5b+8c)^2)/(b^2c)+(10(4a+5b+8c)^2)/(abc)=0 and -(5(4a+5b+8c)^2)/(ac^2)-(4(4a+5b+8c)^2)/(bc^2)+(16(4a+5b+8c)^2)/(abc)=0