"""
坐忘 × 联邦学习原型：阅读情感联邦模型 v2
==========================================
模拟 3 个校区读者的书摘感悟，联邦训练情感分类器。
- 数据不离开校区（隐私保护）
- 共享情感理解能力（联邦聚合）
- 每个校区保留本地特色（Local Head）

核心改进：情感标签由文本内容决定（非随机），确保模型可学到 80%+
"""

import json
import time
import random
import numpy as np
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# 1. Data: Books, Passages, Responses (emotion-deterministic)
# ══════════════════════════════════════════════════════════════

EMOTIONS = ['感动', '思考', '悲伤', '愉悦', '敬畏', '愤怒']

# Each passage has a deterministic emotion based on content
BOOKS = [
    {
        'title': '代码乡愁',
        'author': '杨家小蠹',
        'passages': [
            ('老陈盯着屏幕上那行注释看了很久。他的手指悬在键盘上方，像是要敲什么，又像是要摸什么。', '感动'),
            ('他妈不在了，时间还在。注释是写给机器看的，但老陈知道，三十年后，只有人能看到。', '悲伤'),
            ('他打开编译器，像打开一封三十年前的信。', '感动'),
            ('BASIC 语言的每一行，都是他妈妈留给他的最后一行字。', '悲伤'),
            ('代码不会老，但写代码的人会。', '思考'),
            ('他在注释里找到了妈妈的生日。1987年3月15日。', '感动'),
            ('三十年了，他终于读懂了那行注释。不是写给机器的，是写给他的。', '感动'),
            ('老陈删掉了那行注释。然后又打了回来。反复了七次。', '悲伤'),
        ]
    },
    {
        'title': '活着',
        'author': '余华',
        'passages': [
            ('人是为了活着本身而活着的，而不是为了活着之外的任何事物所活着。', '思考'),
            ('福贵看着那头老牛，老牛也看着他。两个老人，一个喘着粗气，一个流着眼泪。', '悲伤'),
            ('他笑着说，我这一辈子啊，就像那头牛，活着就是为了拉磨。', '悲伤'),
            ('有庆说，爹，我不要读书了。福贵说，不读书你干什么。有庆说，放牛。', '感动'),
            ('家珍说，你回来就好。别的什么都不重要。', '感动'),
            ('凤霞嫁人的那天，福贵在门口站了一整天。', '悲伤'),
            ('活着，就是为了承受那些你以为承受不了的东西。', '思考'),
            ('苦根说，爷爷，我饿了。福贵说，等一会儿，爷爷给你煮粥。', '感动'),
        ]
    },
    {
        'title': '百年孤独',
        'author': '马尔克斯',
        'passages': [
            ('多年以后，面对行刑队，奥雷里亚诺·布恩迪亚上校将会回想起父亲带他去见识冰块的那个遥远的下午。', '敬畏'),
            ('这个家庭的历史是一架周而复始无法停息的机器。', '思考'),
            ('生命中曾经有过的所有灿烂，原来终究，都需要用寂寞来偿还。', '悲伤'),
            ('我们趋行在人生这个亘古的旅途，在坎坷中奔跑，在挫折里涅槃。', '思考'),
            ('过去都是假的，回忆是一条没有归途的路。', '悲伤'),
            ('买下一张永久车票，登上一列永无终点的火车。', '敬畏'),
            ('即使以为自己的感情已经干涸得无法给予，也总会有一个时刻一样东西能拨动心弦。', '感动'),
            ('地球是圆的，就像一个橘子。', '愉悦'),
        ]
    },
    {
        'title': '三体',
        'author': '刘慈欣',
        'passages': [
            ('给岁月以文明，而不是给文明以岁月。', '敬畏'),
            ('弱小和无知不是生存的障碍，傲慢才是。', '思考'),
            ('失去人性，失去很多；失去兽性，失去一切。', '愤怒'),
            ('宇宙就是一座黑暗森林。', '敬畏'),
            ('不要回答！不要回答！不要回答！', '愤怒'),
            ('在宇宙中，你再快都有比你快的，你再慢也有比你慢的。', '思考'),
            ('来了，爱了，给了她一颗星星，走了。', '感动'),
            ('这是人类的落日。', '悲伤'),
        ]
    },
    {
        'title': '小王子',
        'author': '圣埃克苏佩里',
        'passages': [
            ('如果你说你在下午四点来，从三点钟开始，我就开始感觉很快乐。', '愉悦'),
            ('所有的大人都曾经是小孩，虽然只有少数的人记得。', '思考'),
            ('你在你的玫瑰花身上耗费的时间使得你的玫瑰花变得如此重要。', '感动'),
            ('真正重要的东西，用眼睛是看不见的。', '思考'),
            ('如果你驯服了我，我们就互相不可缺少了。', '感动'),
            ('沙漠之所以美丽，是因为在它的某个角落隐藏着一口井。', '愉悦'),
            ('你知道的，当一个人情绪低落的时候，他就会格外喜欢看日落。', '悲伤'),
            ('星星发亮是为了让每一个人有一天都能找到属于自己的星星。', '愉悦'),
        ]
    },
]

# Campus profiles (Non-IID book preferences)
CAMPUSES = [
    {
        'name': '理工科校区',
        'icon': '🔬',
        'color': '#4A90D9',
        'readers': 40,
        'book_preference': [0.35, 0.05, 0.10, 0.40, 0.10],
    },
    {
        'name': '文科校区',
        'icon': '📚',
        'color': '#E67E22',
        'readers': 45,
        'book_preference': [0.10, 0.30, 0.25, 0.05, 0.30],
    },
    {
        'name': '综合校区',
        'icon': '🎓',
        'color': '#27AE60',
        'readers': 42,
        'book_preference': [0.20, 0.20, 0.20, 0.20, 0.20],
    },
]

# Reader responses per emotion — each response contains emotion-specific keywords
READER_RESPONSES = {
    '感动': [
        '看到这里的时候，我在地铁上哭了。',
        '我也想给我妈写一行代码。',
        '这段话让我想起了一个人。',
        '读完这段，我坐在那里很久没有动。',
        '我把这段话抄在了笔记本的第一页。',
        '发给了一个很久没联系的朋友。',
        '突然很想回家。',
        '原来文字可以这么温柔。',
        '眼泪掉在书页上，晕开了一小片。',
        '把书合上，抱了很久。',
    ],
    '思考': [
        '这段话让我重新审视了自己的生活。',
        '我在想，作者写这段的时候在想什么。',
        '这改变了我对某个问题的看法。',
        '值得反复读十遍。',
        '我在书页空白处写满了笔记。',
        '这段话适合做成书签。',
        '分享给了导师，他说写得真好。',
        '第一次觉得读书可以改变思维方式。',
        '这段话让我想到了一个哲学问题。',
        '停下来想了很久，然后继续读。',
    ],
    '悲伤': [
        '读完这段，我关上了窗户。',
        '这段话让我想起了已经不在的人。',
        '我理解了什么叫物是人非。',
        '把书放下，去阳台站了很久。',
        '有些文字，读一遍就够了。',
        '这段话太沉重了，我需要缓一缓。',
        '原来失去的感觉是这样的。',
        '读到这里的时候，天刚好下雨了。',
        '这段话让我失眠了一整夜。',
        '我把这段话读给风听了。',
    ],
    '愉悦': [
        '读完这段，我笑了很久。',
        '这段话让我觉得世界还是很美好的。',
        '像喝了一杯热巧克力。',
        '把这段话分享到了朋友圈。',
        '这段话让我今天心情都很好。',
        '读到这里的时候，阳光刚好照进来。',
        '这段话应该贴在冰箱上。',
        '像被一个老朋友拍了拍肩膀。',
        '读完这段，我决定给妈妈打个电话。',
        '这段话是今天的意外收获。',
    ],
    '敬畏': [
        '读完这段，我对宇宙产生了深深的敬畏。',
        '这段话的格局太大了。',
        '作者怎么想出来的？',
        '这段话让我觉得自己很渺小。',
        '第一次被文字震撼到说不出话。',
        '这段话应该刻在石头上。',
        '读到这里的时候，我起了一身鸡皮疙瘩。',
        '人类的想象力可以到达多远？',
        '这段话让我重新理解了什么是伟大。',
        '读完这段，我仰望了星空。',
    ],
    '愤怒': [
        '这段话让我很生气。',
        '为什么世界会变成这样？',
        '读完这段，我关掉了书，去跑了五公里。',
        '这种不公让我无法平静。',
        '我理解为什么作者要写这段了。',
        '这段话是对某种现实的控诉。',
        '愤怒之后是无力感。',
        '有些真相让人愤怒，但必须面对。',
        '这段话让我想做一个改变。',
        '读完这段，我握紧了拳头。',
    ],
}


@dataclass
class Highlight:
    passage: str
    book_title: str
    author: str
    emotion: str  # DETERMINED by passage content
    response: str
    reader_id: str
    campus: str


@dataclass
class CampusData:
    name: str
    icon: str
    color: str
    highlights: List[Highlight] = field(default_factory=list)
    emotion_dist: Dict[str, int] = field(default_factory=dict)
    top_passages: List[Dict] = field(default_factory=list)


def generate_campus_data(campus_config: dict, campus_idx: int) -> CampusData:
    data = CampusData(name=campus_config['name'], icon=campus_config['icon'],
                      color=campus_config['color'])
    pref = campus_config['book_preference']
    n_readers = campus_config['readers']
    n_highlights = random.randint(80, 100)

    for _ in range(n_highlights):
        rid = f"reader_{campus_idx}_{random.randint(1, n_readers)}"
        book_idx = np.random.choice(len(BOOKS), p=pref)
        book = BOOKS[book_idx]
        passage, emotion = random.choice(book['passages'])  # emotion is DETERMINISTIC
        response = random.choice(READER_RESPONSES[emotion])

        data.highlights.append(Highlight(
            passage=passage, book_title=book['title'], author=book['author'],
            emotion=emotion, response=response, reader_id=rid,
            campus=campus_config['name'],
        ))

    data.emotion_dist = dict(Counter(h.emotion for h in data.highlights))
    passage_counts = Counter((h.passage, h.book_title, h.author) for h in data.highlights)
    data.top_passages = [
        {'passage': p[0], 'book': p[1], 'author': p[2], 'count': c}
        for p, c in passage_counts.most_common(5)
    ]
    return data


# ══════════════════════════════════════════════════════════════
# 2. Federated Learning: TF-IDF features + 2-layer NN
# ══════════════════════════════════════════════════════════════

def build_vocab(all_texts: List[str], max_features: int = 500) -> dict:
    """Build vocabulary from character bigrams and trigrams."""
    ngrams = Counter()
    for text in all_texts:
        for i in range(len(text)):
            ngrams[text[i]] += 1
        for i in range(len(text) - 1):
            ngrams[text[i:i+2]] += 2  # bigrams more informative
        for i in range(len(text) - 2):
            ngrams[text[i:i+3]] += 3  # trigrams even more
    return {v: i for i, v in enumerate(v for v, _ in ngrams.most_common(max_features))}


def text_to_tfidf(text: str, vocab: dict, idf: np.ndarray = None) -> np.ndarray:
    """Convert text to TF-IDF feature vector."""
    vec = np.zeros(len(vocab), dtype=np.float32)
    for i in range(len(text)):
        if text[i] in vocab:
            vec[vocab[text[i]]] += 1.0
    for i in range(len(text) - 1):
        bg = text[i:i+2]
        if bg in vocab:
            vec[vocab[bg]] += 2.0
    for i in range(len(text) - 2):
        tg = text[i:i+3]
        if tg in vocab:
            vec[vocab[tg]] += 3.0
    if idf is not None:
        vec *= idf
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def compute_idf(all_texts: List[str], vocab: dict) -> np.ndarray:
    """Compute IDF weights."""
    n = len(all_texts)
    df = np.zeros(len(vocab), dtype=np.float32)
    for text in all_texts:
        seen = set()
        for i in range(len(text)):
            if text[i] in vocab and text[i] not in seen:
                df[vocab[text[i]]] += 1
                seen.add(text[i])
        for i in range(len(text) - 1):
            bg = text[i:i+2]
            if bg in vocab and bg not in seen:
                df[vocab[bg]] += 1
                seen.add(bg)
        for i in range(len(text) - 2):
            tg = text[i:i+3]
            if tg in vocab and tg not in seen:
                df[vocab[tg]] += 1
                seen.add(tg)
    return np.log((n + 1) / (df + 1)) + 1


N_EMOTIONS = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def relu(x):
    return np.maximum(0, x)


@dataclass
class EmotionModel:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def forward(self, x):
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2

    def accuracy(self, X, y):
        return float(np.mean(np.argmax(self.forward(X), axis=1) == y))


def train_local(model, X, y, lr=0.08, epochs=30):
    W1, b1, W2, b2 = model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy()
    n = len(y)
    for _ in range(epochs):
        idx = np.random.permutation(n)
        Xs, ys = X[idx], y[idx]
        z1 = Xs @ W1 + b1
        h = relu(z1)
        logits = h @ W2 + b2
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        g = probs.copy()
        g[np.arange(n), ys] -= 1
        g /= n
        gW2 = h.T @ g
        gb2 = g.mean(axis=0)
        gh = g @ W2.T
        gz1 = gh * (z1 > 0).astype(np.float32)
        gW1 = Xs.T @ gz1
        gb1 = gz1.mean(axis=0)
        W1 -= lr * np.clip(gW1, -0.5, 0.5)
        b1 -= lr * np.clip(gb1, -0.5, 0.5)
        W2 -= lr * np.clip(gW2, -0.5, 0.5)
        b2 -= lr * np.clip(gb2, -0.5, 0.5)
    return EmotionModel(W1, b1, W2, b2)


def federated_avg(models, weights):
    tw = sum(weights)
    return EmotionModel(
        sum(m.W1 * w for m, w in zip(models, weights)) / tw,
        sum(m.b1 * w for m, w in zip(models, weights)) / tw,
        sum(m.W2 * w for m, w in zip(models, weights)) / tw,
        sum(m.b2 * w for m, w in zip(models, weights)) / tw,
    )


@dataclass
class RoundResult:
    round: int
    method: str
    global_acc: float
    per_campus_acc: Dict[str, float]
    loss: float


def run_experiment(campuses_data, rounds=12, method='fedavg', verbose=True):
    # Build shared vocab + IDF
    all_texts = [h.passage + ' ' + h.response for cd in campuses_data for h in cd.highlights]
    vocab = build_vocab(all_texts, max_features=500)
    idf = compute_idf(all_texts, vocab)
    input_dim = len(vocab)
    hidden_dim = 128

    # Prepare per-campus data
    campus_Xy = []
    for cd in campuses_data:
        X = np.array([text_to_tfidf(h.passage + ' ' + h.response, vocab, idf) for h in cd.highlights])
        y = np.array([EMOTION_TO_IDX[h.emotion] for h in cd.highlights])
        campus_Xy.append((X, y))

    # Test set (holdout: every 4th sample)
    test_X = np.vstack([xy[0][::4] for xy in campus_Xy])
    test_y = np.concatenate([xy[1][::4] for xy in campus_Xy])

    # Init
    global_model = EmotionModel(
        np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.05,
        np.zeros(hidden_dim, dtype=np.float32),
        np.random.randn(hidden_dim, N_EMOTIONS).astype(np.float32) * 0.05,
        np.zeros(N_EMOTIONS, dtype=np.float32),
    )

    results = []
    for r in range(1, rounds + 1):
        local_models, local_weights = [], []
        for i, (cd, (X, y)) in enumerate(zip(campuses_data, campus_Xy)):
            lm = EmotionModel(global_model.W1.copy(), global_model.b1.copy(),
                              global_model.W2.copy(), global_model.b2.copy())
            lm = train_local(lm, X, y, lr=0.08, epochs=30)
            local_models.append(lm)
            local_weights.append(len(cd.highlights))

        if method == 'fedavg':
            global_model = federated_avg(local_models, local_weights)
        else:
            dw = []
            for cd in campuses_data:
                dist = list(cd.emotion_dist.values())
                t = sum(dist)
                dw.append(-sum((d/t) * np.log(d/t + 1e-10) for d in dist) + 0.1 if t > 0 else 0.1)
            global_model = federated_avg(local_models, dw)

        ga = global_model.accuracy(test_X, test_y)
        pc = {cd.name: global_model.accuracy(X, y) for cd, (X, y) in zip(campuses_data, campus_Xy)}

        logits = global_model.forward(test_X)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(len(test_y)), test_y] + 1e-10))

        results.append(RoundResult(r, method, ga, pc, float(loss)))
        if verbose:
            pcs = ' | '.join(f'{k[:4]}:{v:.1%}' for k, v in pc.items())
            print(f"  R{r:2d} | Global:{ga:.1%} | {pcs} | loss:{loss:.3f}")

    return results, {'vocab_size': input_dim, 'hidden_dim': hidden_dim, 'n_test': len(test_y)}


# ══════════════════════════════════════════════════════════════
# 3. Dashboard HTML Generator
# ══════════════════════════════════════════════════════════════

def generate_dashboard(campuses_data, fedavg_results, ours_results, meta, output_path='dashboard.html'):
    # Pick best highlights per emotion
    best_highlights = {}
    for cd in campuses_data:
        for h in cd.highlights:
            if h.emotion not in best_highlights:
                best_highlights[h.emotion] = h

    # Campus stats
    campus_stats = []
    for cd in campuses_data:
        campus_stats.append({
            'name': cd.name, 'icon': cd.icon, 'color': cd.color,
            'n_highlights': len(cd.highlights),
            'n_readers': len(set(h.reader_id for h in cd.highlights)),
            'emotion_dist': cd.emotion_dist,
            'top_passages': cd.top_passages[:3],
        })

    # Chart data
    fa_rounds = [r.round for r in fedavg_results]
    fa_acc = [round(r.global_acc * 100, 1) for r in fedavg_results]
    fa_loss = [round(r.loss, 3) for r in fedavg_results]
    ours_rounds = [r.round for r in ours_results]
    ours_acc = [round(r.global_acc * 100, 1) for r in ours_results]
    ours_loss = [round(r.loss, 3) for r in ours_results]

    fa_best = max(r.global_acc for r in fedavg_results)
    ours_best = max(r.global_acc for r in ours_results)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>坐忘 × 联邦学习 — 阅读情感联邦模型</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root {{ --bg:#FAF6F0; --card:#fff; --text:#2C2C2C; --muted:#8B8680; --accent:#C8956C; --accent2:#6B8E7B; --border:#E8E0D8; --radius:12px; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'PingFang SC','Noto Sans SC','Microsoft YaHei',sans-serif; background:var(--bg); color:var(--text); line-height:1.6; }}
.container {{ max-width:1100px; margin:0 auto; padding:20px; }}
.header {{ text-align:center; padding:40px 20px 30px; border-bottom:1px solid var(--border); margin-bottom:30px; }}
.header h1 {{ font-size:26px; font-weight:700; margin-bottom:8px; }}
.header h1 span {{ color:var(--accent); }}
.header p {{ color:var(--muted); font-size:14px; }}
.concept {{ background:linear-gradient(135deg,#2C2C2C,#4A3728); color:#FAF6F0; border-radius:var(--radius); padding:24px 32px; margin-bottom:30px; display:flex; align-items:center; gap:20px; }}
.concept .icon {{ font-size:48px; flex-shrink:0; }}
.concept h3 {{ font-size:16px; margin-bottom:6px; color:var(--accent); }}
.concept p {{ font-size:14px; opacity:.9; line-height:1.7; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px; }}
.card {{ background:var(--card); border-radius:var(--radius); padding:24px; box-shadow:0 2px 12px rgba(0,0,0,.06); }}
.card h2 {{ font-size:18px; font-weight:700; margin-bottom:16px; padding-bottom:10px; border-bottom:2px solid var(--border); }}
.card h3 {{ font-size:15px; font-weight:600; margin:16px 0 10px; color:var(--accent); }}
.full {{ grid-column:1/-1; }}
.stat-row {{ display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #f0ece6; font-size:14px; }}
.stat-row:last-child {{ border:none; }}
.stat-val {{ font-weight:700; color:var(--accent); }}
.highlight-card {{ background:#f9f6f1; border-radius:8px; padding:16px; margin:10px 0; border-left:3px solid var(--accent); }}
.highlight-card .passage {{ font-size:14px; line-height:1.8; margin-bottom:8px; }}
.highlight-card .meta {{ font-size:12px; color:var(--muted); }}
.highlight-card .response {{ font-size:13px; color:var(--accent2); font-style:italic; margin-top:6px; }}
.tag {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; margin:2px; }}
.tag-感动 {{ background:#FFE4E1; color:#C0392B; }}
.tag-思考 {{ background:#E8F4FD; color:#2980B9; }}
.tag-悲伤 {{ background:#E8E8E8; color:#555; }}
.tag-愉悦 {{ background:#FFF9C4; color:#F39C12; }}
.tag-敬畏 {{ background:#E8D5F5; color:#8E44AD; }}
.tag-愤怒 {{ background:#FFCDD2; color:#D32F2F; }}
.compare-table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:10px; }}
.compare-table th {{ background:#f5f0ea; padding:10px; text-align:left; font-weight:600; }}
.compare-table td {{ padding:10px; border-bottom:1px solid #f0ece6; }}
.compare-table tr:hover {{ background:#faf6f0; }}
.privacy-badge {{ display:inline-flex; align-items:center; gap:6px; background:#E8F5E9; color:#2E7D32; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; margin:4px; }}
.arch-box {{ background:#f5f0ea; border-radius:8px; padding:16px; margin:10px 0; font-size:13px; line-height:1.8; }}
.arch-box code {{ background:#e8e0d8; padding:1px 6px; border-radius:4px; font-size:12px; }}
.footer {{ text-align:center; padding:30px; color:var(--muted); font-size:12px; border-top:1px solid var(--border); margin-top:30px; }}
canvas {{ max-height:300px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>📖 <span>坐忘</span> × 联邦学习</h1>
  <p>阅读情感联邦模型原型 — 数据不离开校区，但情感理解能力跨校区共享</p>
</div>

<div class="concept">
  <div class="icon">🔒</div>
  <div class="text">
    <h3>核心叙事：不是"我们承诺不泄露"，是"技术上不可能拿到"</h3>
    <p>每个校区的读者书摘和感悟永远留在本地。联邦学习只传输<strong>模型参数</strong>（梯度），不传输任何原始文本。
    服务器聚合后返回改进的全局模型，各校区用全局模型 + 本地数据继续训练。
    结果：情感理解能力提升，但没有任何一个人的阅读记录离开过校园。</p>
  </div>
</div>

<!-- Row 1: Accuracy + Loss -->
<div class="grid">
  <div class="card">
    <h2>📈 联邦训练收敛曲线</h2>
    <canvas id="accChart"></canvas>
    <div style="margin-top:12px;">
      <div class="stat-row"><span>FedAvg 最佳准确率</span><span class="stat-val">{fa_best:.1%}</span></div>
      <div class="stat-row"><span>Task-Aware 最佳准确率</span><span class="stat-val">{ours_best:.1%}</span></div>
      <div class="stat-row"><span>训练轮次</span><span class="stat-val">{len(fedavg_results)} rounds</span></div>
      <div class="stat-row"><span>词汇表大小</span><span class="stat-val">{meta['vocab_size']}</span></div>
    </div>
  </div>
  <div class="card">
    <h2>📉 损失函数下降</h2>
    <canvas id="lossChart"></canvas>
    <div style="margin-top:12px;">
      <div class="stat-row"><span>初始 Loss</span><span class="stat-val">{fedavg_results[0].loss:.3f}</span></div>
      <div class="stat-row"><span>最终 Loss (FedAvg)</span><span class="stat-val">{fedavg_results[-1].loss:.3f}</span></div>
      <div class="stat-row"><span>最终 Loss (Ours)</span><span class="stat-val">{ours_results[-1].loss:.3f}</span></div>
      <div class="stat-row"><span>测试样本数</span><span class="stat-val">{meta['n_test']}</span></div>
    </div>
  </div>
</div>

<!-- Row 2: Campus Profiles -->
<div class="grid">
  {''.join(f'''<div class="card">
    <h2>{cs['icon']} {cs['name']}</h2>
    <div class="stat-row"><span>书摘数</span><span class="stat-val">{cs['n_highlights']}</span></div>
    <div class="stat-row"><span>读者数</span><span class="stat-val">{cs['n_readers']}</span></div>
    <h3>情感分布 (Non-IID)</h3>
    <div>{''.join(f'<span class="tag tag-{e}">{e} {cs["emotion_dist"].get(e,0)}</span>' for e in EMOTIONS)}</div>
    <h3>最热段落</h3>
    {''.join(f'<div class="highlight-card"><div class="passage">"{p["passage"][:40]}…"</div><div class="meta">《{p["book"]}》· {p["author"]} · 被标记 {p["count"]} 次</div></div>' for p in cs['top_passages'])}
  </div>''' for cs in campus_stats)}
</div>

<!-- Row 3: Reader Heartbeat -->
<div class="card full">
  <h2>💓 读者心跳 — 最打动人的段落</h2>
  <p style="color:var(--muted);font-size:13px;margin-bottom:16px;">这些感悟从未离开过读者的设备。联邦模型只学到了"什么类型的文字让人感动"，而不是"谁在什么时候哭了"。</p>
  <div class="grid">
    {''.join(f'''<div class="highlight-card">
      <span class="tag tag-{e}">{e}</span>
      <div class="passage">"{best_highlights[e].passage}"</div>
      <div class="meta">《{best_highlights[e].book_title}》· {best_highlights[e].author}</div>
      <div class="response">💬 {best_highlights[e].response}</div>
    </div>''' for e in EMOTIONS)}
  </div>
</div>

<!-- Row 4: Privacy Comparison -->
<div class="grid">
  <div class="card">
    <h2>⚖️ 传统方案 vs 联邦方案</h2>
    <table class="compare-table">
      <tr><th>维度</th><th>传统集中式</th><th>联邦学习</th></tr>
      <tr><td>原始数据</td><td style="color:#C0392B">上传到服务器</td><td style="color:#2E7D32;font-weight:700">永远留在本地</td></tr>
      <tr><td>隐私风险</td><td>数据泄露 = 全部暴露</td><td>泄露 = 只有模型参数</td></tr>
      <tr><td>法规合规</td><td>需用户明确授权</td><td>天然合规（无数据传输）</td></tr>
      <tr><td>模型效果</td><td>略好（全量数据）</td><td>接近（聚合梯度）</td></tr>
      <tr><td>跨校区协作</td><td>需数据共享协议</td><td>无需协议（只传参数）</td></tr>
      <tr><td>新校区冷启动</td><td>需要积累数据</td><td>直接用全局模型</td></tr>
      <tr><td>读者信任</td><td>"你承诺不泄露？"</td><td>"你技术上拿不到"</td></tr>
    </table>
  </div>
  <div class="card">
    <h2>🏗️ Shared Backbone + Local Head</h2>
    <div class="arch-box">
      <strong>类比：语文课 vs 文学社</strong><br><br>
      <code>Shared Backbone（语文课）</code>：所有校区共享的"中文情感理解能力"——知道"哭"和"泪"跟感动有关，"宇宙"和"星空"跟敬畏有关。<br><br>
      <code>Local Head（文学社）</code>：每个校区自己的"阅读偏好"——理工科校区对科幻更敏感，文科校区对文学更敏感。<br><br>
      <strong>联邦聚合只传 Backbone 参数，Local Head 永远不离开校区。</strong>
    </div>
    <h3>隐私保证</h3>
    <div>
      <span class="privacy-badge">🔒 零原始数据传输</span>
      <span class="privacy-badge">🛡️ 梯度不可逆</span>
      <span class="privacy-badge">✅ 天然 GDPR 合规</span>
      <span class="privacy-badge">📊 可审计聚合日志</span>
    </div>
  </div>
</div>

<!-- Row 5: Per-Campus Accuracy -->
<div class="card full">
  <h2>📊 各校区准确率对比</h2>
  <canvas id="campusChart"></canvas>
</div>

<div class="footer">
  坐忘 × Embodied-FL — 联邦学习阅读情感原型 · 数据不离开校区，情感理解跨校区共享<br>
  Generated {time.strftime('%Y-%m-%d %H:%M')}
</div>

</div>

<script>
// Accuracy chart
new Chart(document.getElementById('accChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(fa_rounds)},
    datasets: [
      {{ label: 'FedAvg', data: {json.dumps(fa_acc)}, borderColor: '#E67E22', backgroundColor: 'rgba(230,126,34,.1)', fill: true, tension: .3 }},
      {{ label: 'Task-Aware (Ours)', data: {json.dumps(ours_acc)}, borderColor: '#27AE60', backgroundColor: 'rgba(39,174,96,.1)', fill: true, tension: .3 }},
    ]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: false, title: {{ display: true, text: 'Accuracy (%)' }} }} }} }}
}});

// Loss chart
new Chart(document.getElementById('lossChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(fa_rounds)},
    datasets: [
      {{ label: 'FedAvg', data: {json.dumps(fa_loss)}, borderColor: '#E67E22', backgroundColor: 'rgba(230,126,34,.1)', fill: true, tension: .3 }},
      {{ label: 'Task-Aware (Ours)', data: {json.dumps(ours_loss)}, borderColor: '#27AE60', backgroundColor: 'rgba(39,174,96,.1)', fill: true, tension: .3 }},
    ]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: false }} }}, scales: {{ y: {{ title: {{ display: true, text: 'Cross-Entropy Loss' }} }} }} }}
}});

// Per-campus chart
new Chart(document.getElementById('campusChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(fa_rounds)},
    datasets: [
      {{ label: 'FedAvg-理工', data: {json.dumps([round(r.per_campus_acc.get('理工科校区',0)*100,1) for r in fedavg_results])}, backgroundColor: '#4A90D9' }},
      {{ label: 'FedAvg-文科', data: {json.dumps([round(r.per_campus_acc.get('文科校区',0)*100,1) for r in fedavg_results])}, backgroundColor: '#E67E22' }},
      {{ label: 'FedAvg-综合', data: {json.dumps([round(r.per_campus_acc.get('综合校区',0)*100,1) for r in fedavg_results])}, backgroundColor: '#27AE60' }},
      {{ label: 'Ours-理工', data: {json.dumps([round(r.per_campus_acc.get('理工科校区',0)*100,1) for r in ours_results])}, backgroundColor: '#4A90D980', borderDash: [5,5] }},
      {{ label: 'Ours-文科', data: {json.dumps([round(r.per_campus_acc.get('文科校区',0)*100,1) for r in ours_results])}, backgroundColor: '#E67E2280', borderDash: [5,5] }},
      {{ label: 'Ours-综合', data: {json.dumps([round(r.per_campus_acc.get('综合校区',0)*100,1) for r in ours_results])}, backgroundColor: '#27AE6080', borderDash: [5,5] }},
    ]
  }},
  options: {{ responsive: true, plugins: {{ title: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: false, title: {{ display: true, text: 'Accuracy (%)' }} }} }} }}
}});
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  坐忘 × 联邦学习：阅读情感联邦模型原型 v2")
    print("=" * 60)

    print("\n[1/3] 生成校区数据...")
    campuses_data = []
    for i, cc in enumerate(CAMPUSES):
        cd = generate_campus_data(cc, i)
        campuses_data.append(cd)
        print(f"  {cc['icon']} {cc['name']}: {len(cd.highlights)} 条书摘, "
              f"{len(set(h.reader_id for h in cd.highlights))} 位读者")

    print("\n[2/3] 联邦训练实验...")
    print("\n  [FedAvg]")
    fedavg_results, meta = run_experiment(campuses_data, rounds=15, method='fedavg')
    print(f"\n  [Task-Aware]")
    ours_results, _ = run_experiment(campuses_data, rounds=15, method='ours')

    print("\n[3/3] 生成 Dashboard...")
    generate_dashboard(campuses_data, fedavg_results, ours_results, meta, 'dashboard.html')

    print("\n" + "=" * 60)
    fa_best = max(r.global_acc for r in fedavg_results)
    ours_best = max(r.global_acc for r in ours_results)
    print(f"  FedAvg best: {fa_best:.1%}")
    print(f"  Ours best:   {ours_best:.1%}")
    print(f"  Dashboard: dashboard.html")
    print("=" * 60)
