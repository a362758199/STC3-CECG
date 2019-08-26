import random
from emotional_classifier import five_emotional_classifier


def load_all_classifier():
    global dict_text1, ecm1, dict_text2, ecm2, dict_text3, ecm3, dict_text4, ecm4, dict_text5, ecm5
    dict_text1, ecm1 = five_emotional_classifier.load_classifier('train_emotion1_neg.txt', 'train_emotion1_pos.txt',
                                                                 'emotion1_classification_model.h5')
    dict_text2, ecm2 = five_emotional_classifier.load_classifier('train_emotion2_neg.txt', 'train_emotion2_pos.txt',
                                                                 'emotion2_classification_model.h5')
    dict_text3, ecm3 = five_emotional_classifier.load_classifier('train_emotion3_neg.txt', 'train_emotion3_pos.txt',
                                                                 'emotion3_classification_model.h5')
    dict_text4, ecm4 = five_emotional_classifier.load_classifier('train_emotion4_neg.txt', 'train_emotion4_pos.txt',
                                                                 'emotion4_classification_model.h5')
    dict_text5, ecm5 = five_emotional_classifier.load_classifier('train_emotion5_neg.txt', 'train_emotion5_pos.txt',
                                                                 'emotion5_classification_model.h5')
    return dict_text1, ecm1, dict_text2, ecm2, dict_text3, ecm3, dict_text4, ecm4, dict_text5, ecm5
    print('load successfully')

print('generate_method_ntcir')
dict_text1, ecm1, dict_text2, ecm2, dict_text3, ecm3, dict_text4, ecm4, dict_text5, ecm5 = load_all_classifier()


def count_emotion_score(text):
    emotion_score = []
    emotion_score.append(five_emotional_classifier.predict(text, dict_text1, ecm1))
    emotion_score.append(five_emotional_classifier.predict(text, dict_text2, ecm2))
    emotion_score.append(five_emotional_classifier.predict(text, dict_text3, ecm3))
    emotion_score.append(five_emotional_classifier.predict(text, dict_text4, ecm4))
    emotion_score.append(five_emotional_classifier.predict(text, dict_text5, ecm5))
    # emotion_score = [float(f) for f in emotion_score]
    return emotion_score


# print('emotion_label=', emotion_score.index(max(emotion_score))+1)


# load_all_classifier()


# ######------------NTCIR-------------#########


def ntcir_ecs(result):  # emotion classification subsystem
    default_sentence = {
        1: random.choice(["是 的 ， 我 也 是",
                          "哈哈 ， 我 也 觉得",
                          "哈哈 ， 我 也 是",
                          "来 抱抱 [ 噢 耶 ]",
                          "顺其自然 好 了 [ 呵呵 ]"]),
        2: random.choice(["我 也 是 [ 泪 ]",
                          "我 错 了 ，我 错 了",
                          "悲 摧 啊",
                          "烦 烦 烦 烦 烦 ",
                          "唉 ， 我 也 觉得 "]),
        3: random.choice(["讨厌 讨厌 讨厌",
                          "真的 假 的 ?",
                          "什么 情况 ?",
                          "是 啊 ， 我 要 哭 了 。",
                          "[ 挖 鼻 屎 ]"]),
        4: random.choice(["什么 情况 ? [ 怒 ]",
                          "[ 怒 ]",
                          " 又 咋 了 ? ?",
                          "什么 情况 ? [ 怒 ]",
                          "什么 情况 ? [ 怒 ]"]),
        5: random.choice(["哈哈 ， 你 也 是",
                          "顺其自然 好 了 [ 呵呵 ]",
                          "真的 假 的 ?",
                          "哈哈 哈哈 哈 哈 哈哈 哈哈 哈哈 哈哈 哈 哈 哈哈 哈哈 哈哈 哈哈 哈 哈",
                          "[ 哈哈 ] [ 哈 哈 ] [ 哈哈 ]"])
    }
    # cesl = [[], [], [], [], []]  # 5*5  candidate_emotion_score_list
    emo = {}
    emo[1] = []
    emo[2] = []
    emo[3] = []
    emo[4] = []
    emo[5] = []
    for i in range(0, 5):
        sentr = result[i]
        #print(sentr)
        score = count_emotion_score(sentr)
        score = [(float(s[0]),s[1]) for s in score]
        #print(score)
        emo[1].append([sentr, score[0][0]])
        emo[2].append([sentr, score[1][0]])
        emo[3].append([sentr, score[2][0]])
        emo[4].append([sentr, score[3][0]])
        emo[5].append([sentr, score[4][0]])

    emo[1] = sorted(emo[1], key=lambda x: x[1], reverse=True)
    emo[2] = sorted(emo[2], key=lambda x: x[1], reverse=True)
    emo[3] = sorted(emo[3], key=lambda x: x[1], reverse=True)
    emo[4] = sorted(emo[4], key=lambda x: x[1], reverse=True)
    emo[5] = sorted(emo[5], key=lambda x: x[1], reverse=True)
    #print(emo[1], emo[2], emo[3], emo[4], emo[5], sep = '\n')
    for n in range(1, 6):
        if emo[n][0][1] < 0.5:  # threshold = 0.5
            emo[n][0][0] = default_sentence[n]
        else:
            pass
    maxsentences = [emo[i][0][0] for i in range(1,6)]
    return maxsentences



#candidata_results = ['漂亮 的 花 ', '我 也 觉得 很 美 很 美 很 美 ', '我 也 觉得 很 美 哦 。 。 。 ', '我 也 觉得 很 美 很 美 。 。 。 ', '我 也 觉得 很 美 很 美 ']
#print(ntcir_ecs(candidata_results))
#print(cgzy_cm('漂亮 的 花 '))
count_emotion_score('漂亮 的 花 ')