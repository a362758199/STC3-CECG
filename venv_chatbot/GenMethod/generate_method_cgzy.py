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
    print('load successfully ')


print('generate_method_cgzy')
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


#print(count_emotion_score('好 漂亮 的 花'))


# ######------------CGZY-------------#########
def cgzy_cm(text):  # cgzy choose model
    # esl = []
    num2modelname = {
        1: 'train_emotion1_chinyi',
        2: 'train_emotion2_shihhsiang',
        3: 'train_emotion3_zhongyi',
        4: 'train_emotion4_yijing',
        5: 'train_emotion5_kunli'
    }
    emo_match_table = {
        1: random.choice([1, 3, 5]),
        2: random.choice([2, 3, 5]),
        3: random.choice([1, 3, 5]),
        4: random.choice([2, 3, 5]),
        5: random.choice([1, 3, 5])
    }
    # print('({})'.format(text))
    print(text,end = '\t')
    esl = count_emotion_score(text)  # emotion_score_list
    esl = [float(f[0]) for f in esl]
    # print(esl)
    label = esl.index(max(esl)) + 1
    print(label, end = '\t')
    modelNo = emo_match_table[label]
    print(modelNo)
    input = {
        'model': num2modelname[modelNo],
        'epoch': 5000,
        'topn': 1,
        'query': text
    }
    return input


# cgzy_cm('好 漂亮 的 花' )
tdpo = ["今年 的 实习 生 , 嗯 , 还 挺 帅 的 。 。 。","很 想 吃   多力 多 滋","保佑 保佑 , 一 次 过 [ 给 力 ] [ 给 力 ] [ 给 力 ]","每天 都 有 旺仔 喝 ~ 爽 ! !","他家 的 饭 做 的 还是 很 不错 。 [ 馋嘴 ]","最近 大爱 adele 的 歌 ~ ~ ~",
"已经 冷 无可 冷 了 ! ! ! 求 温暖 ~","心烦 的 事 总是 成 双 , 计划 总是 赶 不 上 变化 。 两难 。","还是 从 前 那个 丢三落四 的 我 !","我 也 想 对 全 世界 说 昨晚 对不起 .","我 想 说 : 我 开始 对 你 失望 了 · · · · · ·","真 倒霉 , 上午 被 领导 骂 , 下午 喝水 把 嘴唇 磕 破 了 。",
"郑州 的 出租 车 , 太 没 道德 性 了 .","打 了 半 下午 球 , 脚 断 了 一样 !","团购 的 待遇 往往 很 差 。 [ 汗 ]","很 怕 被 人 搭讪 , 太 恐怖 了 。 。 。","摔 了 个 大 跟头 。 震 得 头 好 晕 [ 晕 ]","foxmail 和 微 信 今天 都 出 问题 , 讨厌 [ 抓狂 ]",
"居然 这 时候 刮 台风 ! ! ! !  我 恨 你 ! ! ! !","我 想 把 你 砸 了   打卡 机","吵架 吵 到 神经 痛 [ 怒 ] [ 抓狂 ] [ 顶 ] [ 怒骂 ] [ 鄙视 ]","大 清早 起来 就 气 不顺 ! 今天 不 爽 !","可恶 的 垃圾 短信 [ 怒 ]","打 个 麻将 都 找 不到 人 [ 怒骂 ]",
"一个 人 唱 了 4 小时 , 我 表示 我 瘦 了 [ 偷 笑 ]","直播 + 电视   那 是 相当 给 力 !   [ 哈哈 ]   [ 哈哈 ]   [ 哈哈 ]","睡 得 好 舒服 [ 酷 ] 连续 两 天 睡 一个 下午","我 要 订婚 了 同志 们 ! 哈 哈哈","周五 拿 驾照 去 [ 哈哈 ] [ 哈哈 ] [ 哈哈 ]","帝都 春天 好多 雨 [ 鼓掌 ]"
]
for s in tdpo:
    cgzy_cm(s)
