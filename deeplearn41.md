---
title: jieba分词的奇妙
date: 2026-10-14 21:27:38
tags:
---

### jieba分词的解释
这段代码的作用是：**对 DataFrame 中“内容”这一列的每条用户评论做中文分词，并把每条评论分出来的词用英文逗号拼接成一个字符串，最后把所有评论组成一个列表。**

逐行解释：

```python
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
```

等价于：

```python
comment_list = []
for line in df['内容']:
    words = jieba.lcut(line)      # 对一条评论分词，返回词列表
    comment = ','.join(words)     # 用逗号把词连接成一个字符串
    comment_list.append(comment)  # 加入结果列表
```

#### 1. `df['内容']`

表示从 DataFrame `df` 中取出“内容”这一列，也就是所有用户评论。

例如：

```python
df['内容']
```

可能是：

```python
0    这个商品很好用
1    物流太慢了
2    质量不错，下次还买
Name: 内容, dtype: object
```

#### 2. `for line in df['内容']`

逐条遍历每一条评论。每次循环中，`line` 是一条评论文本。

#### 3. `jieba.lcut(line)`

使用 `jieba` 对一条评论进行中文分词，返回一个词列表。

例如：

```python
jieba.lcut('这个商品很好用')
```

可能得到：

```python
['这个', '商品', '很', '好用']
```

注意：`jieba.lcut()` 直接返回列表，适合后面继续处理。

#### 4. `','.join(...)`

把分词得到的词列表用英文逗号连接成一个字符串。

例如：

```python
','.join(['这个', '商品', '很', '好用'])
```

得到：

```python
'这个,商品,很,好用'
```

#### 5. 列表推导式

```python
[','.join(jieba.lcut(line)) for line in df['内容']]
```

表示对每一条评论都执行“分词 + 用逗号拼接”，最后得到一个新的列表。

例如原始评论是：

```python
[
    '这个商品很好用',
    '物流太慢了',
    '质量不错，下次还买' 
]
```

分词拼接后可能变成：

```python
[
    '这个,商品,很,好用',
    '物流,太,慢,了',
    '质量,不错,，,下次,还,买'
]
```

所以：

```python
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
```

最终 `comment_list` 的数据形式是：

```python
[
    '第1条评论切词1,切词2,切词3...',
    '第2条评论切词1,切词2,切词3...',
    ...
]
```

#### 6. `print(comment_list)`

打印分词后的评论列表，方便查看结果。
---








# 从 jieba 分词到词频矩阵：停用词加载与 CountVectorizer 向量化

在文本分类、情感分析等任务里，中文评论的处理链路通常是：

```text
原始评论 → jieba 分词 → 去停用词 → 词频向量化 → 喂给模型
```

---

## 一、整体流程先看一眼

```text
原始评论 df['内容']
        ↓ 2.4  jieba 分词 + 逗号拼接
comment_list = ['这个,商品,很,好用', '物流,太,慢,了', ...]
        ↓ 2.5  加载停用词
stopwords_list = ['的', '了', '啊', ...]
        ↓ 2.6  创建 CountVectorizer(stop_words=...)
        ↓ 2.7  fit_transform + toarray
x = 词频矩阵
```

最终得到：

```python
x = [
    [词1次数, 词2次数, 词3次数, ...],   # 第1条评论
    [词1次数, 词2次数, 词3次数, ...],   # 第2条评论
    ...
]
```

下面逐段展开。

---

## 二、2.5 加载停用词列表

```python
with open('./data/stopwords.txt', 'r', encoding='utf-8') as src_f:
    # 2.5.1 一次读取所有的行
    stopwords_list = src_f.readlines()
    # 2.5.2 删除最后的 '\n'
    stopwords_list = [line.strip() for line in stopwords_list]
    # 2.5.3 对停用词列表去重
    stopwords_list = list(set(stopwords_list))
    print(stopwords_list)
```

### 什么是停用词？

停用词是**对文本语义贡献很小、通常需要被删除的词**，例如：

```text
的、了、啊、哈、从、都、是、在、和、就、也、很……
```

这些词几乎在每条评论里都会出现，对区分评论内容帮助不大，保留下来反而会增加维度、干扰模型，所以一般要过滤掉。

### 2.5.1 一次读取所有行

```python
stopwords_list = src_f.readlines()
```

`readlines()` 会把文件每一行读成一个字符串，返回一个列表。

假设 `stopwords.txt` 内容是：

```text
的
了
啊
哈
从
都
```

则得到：

```python
['的\n', '了\n', '啊\n', '哈\n', '从\n', '都\n']
```

注意：每行末尾都带着换行符 `\n`。

### 2.5.2 删除每行末尾的 `\n`

```python
stopwords_list = [line.strip() for line in stopwords_list]
```

`strip()` 会删除字符串首尾的空白字符，包括空格、`\n`、`\t` 等。

结果变成：

```python
['的', '了', '啊', '哈', '从', '都']
```

这一步很关键：如果不删 `\n`，停用词就变成 `"的\n"`，而评论分词后得到的是 `"的"`，两者不相等，停用词就过滤不掉。

### 2.5.3 去重

```python
stopwords_list = list(set(stopwords_list))
```

`set()` 会去掉重复元素，因为停用词文件里可能有重复行。

例如：

```python
['的', '了', '的', '啊']
```

变成：

```python
{'的', '了', '啊'}
```

再转回列表：

```python
['的', '了', '啊']
```

注意：`set()` 是**无序**的，所以去重后列表顺序会变化，但这不影响 `CountVectorizer` 使用。

### 2.5.4 打印

```python
print(stopwords_list)
```

查看加载到的停用词列表。

---

## 三、2.6 创建向量化对象

```python
transfer = CountVectorizer(stop_words=stopwords_list)
```

`CountVectorizer` 是 `sklearn.feature_extraction.text` 中的工具，作用是：

1. 把文本切分成词；
2. 过滤掉停用词；
3. 统计每个词在每条文本中出现的次数；
4. 生成词频矩阵。

### 关键参数 `stop_words`

```python
CountVectorizer(stop_words=stopwords_list)
```

`stop_words` 指定停用词列表，`CountVectorizer` 在统计词频时会**自动忽略这些词**。

也就是说，这里把 2.5 加载的停用词交给向量化器，由它来完成删除停用词的工作。

### 注意：`CountVectorizer` 默认分词规则

`CountVectorizer` 默认的分词器 `token_pattern=r"(?u)\b\w\w+\b"` 会：

- 按空格和标点切分；
- **只保留长度 ≥ 2 的词**；
- 默认把输入当成“一个字符串”，而不是词列表。

你前面传给它的 `comment_list` 是这样的：

```python
['这个,商品,很,好用', '物流,太,慢,了']
```

`CountVectorizer` 会把逗号当作分隔符，把它切成：

```python
['这个', '商品', '很', '好用']
```

但是注意：**单字词（如“很”“了”“太”）默认会被 `token_pattern` 过滤掉**，因为默认只匹配长度 ≥ 2 的词。

如果你的停用词表里有单字词（如“的”“了”“啊”），它们本来就可能已经被 `token_pattern` 过滤掉了，所以停用词过滤主要对长度 ≥ 2 的词生效。

如果要保留单字词，需要修改参数，例如：

```python
CountVectorizer(
    stop_words=stopwords_list,
    token_pattern=r"(?u)\b\w+\b"   # 保留单字
)
```

或者直接传分词后的词列表（见文末“常见疑问”部分）。

---

## 四、2.7 统计词频矩阵

```python
# transfer.fit(comment_list)
# x = transfer.transform(comment_list).toarray()
x = transfer.fit_transform(comment_list).toarray()
print(x)
```

### `fit`、`transform`、`fit_transform` 的区别

```python
transfer.fit(comment_list)                       # 只学习词表
x = transfer.transform(comment_list).toarray()   # 按学到的词表转换
```

等价于：

```python
x = transfer.fit_transform(comment_list).toarray()
```

- `fit`：学习所有出现的词，建立“词表”，并统计文档频率，用于过滤停用词等。
- `transform`：把每条文本转成向量。
- `fit_transform`：先 `fit` 再 `transform`，一步完成。

### `toarray()` 的作用

`transform()` 返回的是**稀疏矩阵**（`scipy.sparse`），因为文本词频矩阵通常非常稀疏（大量 0）。

```python
x = transfer.fit_transform(comment_list)
```

得到的是稀疏矩阵。

```python
.toarray()
```

把它转成普通的 NumPy 二维数组，方便查看和后续处理。

### 矩阵 `x` 的格式

注释里写：

```python
# x的格式: [[第1条评论的切词分布, 有就是1, 没有就是0], [第2条评论的切词分布, 有就是1, 没有就是0], ...]
```

这个描述不太准确。实际上 `x` 是**词频矩阵**，不是 0/1 矩阵。

- 每一行：一条评论；
- 每一列：词表中的一个词；
- 每个元素：该词在该条评论中出现的**次数**。

### 举例

假设 `comment_list` 是：

```python
[
    '这个,商品,很,好用',
    '物流,太,慢,了',
    '质量,不错,下次,还,买'
]
```

`CountVectorizer` 学习到的词表（假设停用词把“这个”“很”等去掉了）可能是：

```python
['商品', '好用', '物流', '太慢', '质量', '不错', '下次']
```

则 `x` 可能是：

```python
[
    [1, 1, 0, 0, 0, 0, 0],   # 第1条评论
    [0, 0, 1, 1, 0, 0, 0],   # 第2条评论
    [0, 0, 0, 0, 1, 1, 1]    # 第3条评论
]
```

每一列对应一个词，每一行是一条评论。

### 如何查看“列名”？

可以调用：

```python
print(transfer.get_feature_names_out())
```

得到词表顺序，例如：

```python
['商品' '好用' '物流' '太慢' '质量' '不错' '下次']
```

这样就能知道 `x` 的每一列代表哪个词。

### 关于 `toarray()` 的提醒

评论数量很大、词表很长时，`toarray()` 会占用大量内存。如果后续要喂给模型，通常直接传稀疏矩阵：

```python
x = transfer.fit_transform(comment_list)
```

只在需要查看或小数据时才 `toarray()`。

---

## 五、常见疑问解答

### 疑问 1：`strip()` 不能直接传 `stopwords_list` 吗？

**不能。** `strip()` 是**字符串的方法**，只能作用在单个字符串上，不能直接作用在列表上。

`stopwords_list` 是一个列表，例如：

```python
stopwords_list = ['的\n', '了\n', '啊\n']
```

如果你写：

```python
stopwords_list.strip()   # ❌ 报错
```

会报：

```text
AttributeError: 'list' object has no attribute 'strip'
```

因为 `list` 没有 `strip` 方法。

所以必须**逐个元素**调用 `strip()`，有几种写法：

#### 写法 1：列表推导式（当前代码用的）

```python
stopwords_list = [line.strip() for line in stopwords_list]
```

意思是：对列表里每个字符串 `line` 都执行 `line.strip()`，结果组成新列表。

#### 写法 2：`map`

```python
stopwords_list = list(map(str.strip, stopwords_list))
```

`map(str.strip, stopwords_list)` 会把 `strip` 依次作用到每个元素上，`list()` 再转回列表。

#### 写法 3：循环

```python
for i in range(len(stopwords_list)):
    stopwords_list[i] = stopwords_list[i].strip()
```

#### 对比

```python
'的\n'.strip()            # ✅ '的'         字符串可以
['的\n', '了\n'].strip()  # ❌ AttributeError  列表不行
```

**结论：** 列表没有 `strip()`，必须对每个元素单独调用，所以要用列表推导式或 `map`。

---

### 疑问 2：`list(set(stopwords_list))` 的转换流程是怎样的？

这行代码分三步，从内到外执行：

```python
stopwords_list = list(set(stopwords_list))
```

假设：

```python
stopwords_list = ['的', '了', '的', '啊', '了']
```

#### 第一步：`set(stopwords_list)`

`set()` 把列表转成**集合**。集合的特点是：

- **元素不重复**（自动去重）；
- **无序**（不保证顺序）；
- 只能放可哈希对象（字符串可以）。

结果：

```python
set(stopwords_list)
# 得到：{'的', '了', '啊'}   （顺序不保证）
```

这里 `'的'` 和 `'了'` 各出现两次，被去重成一次。

#### 第二步：`list(...)`

`set` 是集合类型，不能直接当下标用，也不能直接传给需要列表的地方，所以再转回列表：

```python
list(set(stopwords_list))
# 得到：['的', '了', '啊']   （顺序不保证）
```

#### 第三步：赋值

```python
stopwords_list = list(set(stopwords_list))
```

把去重后的列表重新赋给 `stopwords_list`。

#### 完整流程

```text
['的', '了', '的', '啊', '了']
        ↓ set()  去重 + 变成集合
{'的', '了', '啊'}
        ↓ list()  转回列表
['的', '了', '啊']
```

#### 为什么不用别的方式去重？

也可以用：

```python
# 保持顺序的去重（Python 3.7+）
stopwords_list = list(dict.fromkeys(stopwords_list))
```

或者：

```python
stopwords_list = list(set(stopwords_list))
```

两种都可以。`set` 方式最简单，但**会打乱顺序**。对停用词来说，顺序无所谓，因为 `CountVectorizer` 只关心“这个词在不在停用词集合里”，不关心顺序。

#### 注意：`set` 只能去重“可哈希”对象

字符串是可哈希的，所以没问题。如果列表元素是列表（不可哈希），就会报错：

```python
set([['a'], ['b']])   # ❌ TypeError: unhashable type: 'list'
```

这里每个元素都是字符串，所以安全。

---

### 疑问 3：`x = transfer.fit_transform(comment_list)` 这里直接传吗？

**可以直接传，但要看 `comment_list` 是什么格式，以及你期望 `CountVectorizer` 怎么分词。**

你前面的 `comment_list` 是：

```python
comment_list = [','.join(jieba.lcut(line)) for line in df['内容']]
# 结果形如：
# ['这个,商品,很,好用', '物流,太,慢,了']
```

这是**字符串列表**，每个元素是一条“用逗号拼接的词串”。

#### 情况 A：直接传字符串列表（当前代码）

```python
x = transfer.fit_transform(comment_list).toarray()
```

**可以传**，`CountVectorizer` 接受字符串列表。它会：

1. 对每个字符串按 `token_pattern` 再切一次；
2. 逗号会被当作分隔符；
3. 默认 `token_pattern=r"(?u)\b\w\w+\b"`，**只保留长度 ≥ 2 的词**；
4. 过滤 `stop_words` 里的词；
5. 统计词频。

所以它能跑通，但有两个“副作用”：

- **单字词会被丢掉**（如“很”“了”“太”），因为默认只匹配长度 ≥ 2；
- **会按它自己的规则再切一次**，而不是完全复用 jieba 的结果。

如果你能接受这两个副作用，直接传没问题。

#### 情况 B：传“词列表的列表”（更推荐，完全复用 jieba 分词）

如果你希望**完全使用 jieba 的分词结果**，并且自己控制停用词过滤，可以这样：

```python
comment_words = [
    [w for w in jieba.lcut(line) if w not in stopwords_list]
    for line in df['内容']
]
# 结果形如：
# [['这个', '商品', '很', '好用'], ['物流', '太', '慢', '了']]

transfer = CountVectorizer(analyzer=lambda x: x)  # 直接使用传入的词列表
x = transfer.fit_transform(comment_words).toarray()
```

`analyzer=lambda x: x` 的意思是：**不要再切词，直接把输入当成词列表用。**

这样：

- 完全复用 jieba 的分词；
- 单字词也能保留；
- 停用词已经在列表推导式里过滤掉了。

#### 情况 C：直接传字符串列表，但想保留单字词

```python
transfer = CountVectorizer(
    stop_words=stopwords_list,
    token_pattern=r"(?u)\b\w+\b"   # 保留单字
)
x = transfer.fit_transform(comment_list).toarray()
```

#### 三种方式对比

| 方式 | 输入 | 是否再分词 | 单字词 | 停用词过滤 |
|---|---|---|---|---|
| A 当前代码 | 字符串列表 `['这个,商品,很,好用']` | 会，按 `token_pattern` | 默认丢 | 由 `stop_words` 过滤 |
| B 推荐 | 词列表 `[['这个','商品','很','好用']]` | 不会（`analyzer=lambda x: x`） | 保留 | 自己在列表推导式里过滤 |
| C | 字符串列表 | 会 | 保留（改 `token_pattern`） | 由 `stop_words` 过滤 |

#### `fit_transform` 到底能传什么？

`CountVectorizer.fit_transform` 接受的是**可迭代的文档集合**，每个文档可以是：

- 字符串（默认情况）；
- 或者任意可迭代对象（词列表），但前提是你改了 `analyzer`/`tokenizer`。

所以：

```python
x = transfer.fit_transform(comment_list)   # ✅ comment_list 是字符串列表，可以
```

但如果你传的是：

```python
comment_list = [['这个', '商品'], ['物流', '太', '慢']]   # 词列表
x = transfer.fit_transform(comment_list)   # ❌ 默认会报错或行为不对
```

因为默认 `analyzer` 期望的是字符串，对列表调用 `.lower()` 等会出错。这时必须加 `analyzer=lambda x: x`。

---

## 六、几个容易忽略的细节

### 1. `CountVectorizer` 会再切一次词

你前面已经用 `jieba` 分好词并用逗号拼成字符串了，`CountVectorizer` 内部还会按它的 `token_pattern` 再切一次。

- 逗号、空格通常会被当作分隔符；
- 默认只保留长度 ≥ 2 的词；
- 单字词、标点会被过滤掉。

如果你希望**直接使用 jieba 的分词结果**，不让 `CountVectorizer` 再切，就用情况 B 的 `analyzer=lambda x: x` 写法。

### 2. `stop_words` 大小写敏感

`CountVectorizer` 有一个参数 `lowercase=True`（默认），会把所有文本转成小写后再匹配停用词。中文没有大小写问题，但如果文本里混有英文，要注意停用词表也要用对应大小写形式。

### 3. 停用词表要和分词结果“完全一致”

例如停用词文件里是：

```text
的
了
```

分词结果是：

```text
的
了
```

才能匹配上。如果停用词里写成“的了”，或者带空格、带 `\n`，就匹配不上。所以 2.5.2 的 `strip()` 非常重要。

### 4. `x` 是稀疏的，评论多时不要随便 `toarray()`

评论数量很大、词表很长时，`toarray()` 会占用大量内存。如果后续要喂给模型，通常直接传稀疏矩阵：

```python
x = transfer.fit_transform(comment_list)
```

只在需要查看或小数据时才 `toarray()`。

---

## 七、总结

- **2.5**：从 `stopwords.txt` 读取停用词，去掉换行符，去重，得到停用词列表。
- **2.6**：创建 `CountVectorizer`，把停用词列表传进去，让它统计词频时忽略这些词。
- **2.7**：用 `fit_transform` 学习词表并生成词频矩阵，`toarray()` 转成普通二维数组 `x`。
- `x` 的每一行是一条评论，每一列是一个词，值是词在该评论中出现的次数。
- 需要注意 `CountVectorizer` 默认会再次分词、过滤单字词，如果想完全复用 jieba 结果，应自定义 `analyzer`。

### 三个常见疑问的简短答案

1. **`strip()` 不能直接传列表**，因为 `strip` 是字符串方法，列表没有这个方法。要用列表推导式 `[line.strip() for line in ...]` 或 `map(str.strip, ...)` 对每个元素单独处理。

2. **`list(set(stopwords_list))` 流程**：`set()` 把列表转成集合，自动去重、但无序；`list()` 再把集合转回列表。目的是去重，代价是顺序可能变化，对停用词无影响。

3. **`fit_transform(comment_list)` 可以直接传**，因为 `comment_list` 是字符串列表。但 `CountVectorizer` 会按自己的 `token_pattern` 再分词、默认丢单字词。如果想完全复用 jieba 结果，应传词列表并设置 `analyzer=lambda x: x`；如果想保留单字，可改 `token_pattern=r"(?u)\b\w+\b"`。

把这条链路走通，后面的 TF-IDF、朴素贝叶斯、SVM、LDA 等任务就有了干净的输入。