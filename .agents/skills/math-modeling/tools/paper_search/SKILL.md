---
name: "paper-search"
description: "Search academic papers via OpenAlex API for math modeling references. Invoke when user needs literature search, paper references, or when writing papers requires citations."
---

# Paper Search Skill - 论文搜索技能

本技能通过 **OpenAlex API** 实现学术论文搜索功能，为数学建模论文撰写提供英文/国际文献的初筛支持。

> 注意：OpenAlex 不等同于完整中文文献检索。涉及中文文献、知网、万方、维普或核心期刊筛选时，优先使用已安装的 `lit-tracker` 能力，或要求用户提供数据库导出、截图、链接、PDF 等可验证材料。

---

## 功能概述

| 功能 | 说明 |
|------|------|
| **论文搜索** | 通过关键词搜索相关学术论文 |
| **摘要获取** | 自动重建并返回论文摘要 |
| **引用格式化** | 生成标准的引用格式 |
| **多字段搜索** | 支持标题、摘要、作者等多字段搜索 |

---

## 使用场景

在以下情况下使用本技能：

1. **建模分析阶段**：查找模型相关的英文/国际理论文献
2. **论文撰写阶段**：为论文添加英文/国际参考文献引用
3. **算法验证阶段**：查找算法的原始论文
4. **用户请求**：用户明确要求搜索英文论文或国际文献

中文文献检索请按 `../../references/文献检索与引用规范.md` 执行。

---

## 配置要求

### 邮箱配置（必需）

OpenAlex API 要求提供邮箱地址以使用礼貌池（Polite Pool）：

```
使用前必须配置邮箱地址：
- 用于礼貌池访问，提高API限制
- 必须是有效的邮箱地址
```

---

## 搜索流程

```
┌─────────────────────────────────────────────────────────────┐
│                      论文手/建模手                           │
│  (需要引用文献时调用搜索工具)                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ 调用 search_papers
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 openalex_scholar.py                         │
│  (论文搜索脚本 - 封装 OpenAlex API 调用)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP GET 请求
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenAlex API (https://api.openalex.org)        │
│  - 免费学术搜索引擎                                         │
│  - 支持论文标题、摘要、作者、引用次数等字段                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 使用方法

### 方法一：运行搜索脚本

```bash
python tools/paper_search/scripts/openalex_scholar.py --query "grey prediction model" --email "your@email.com"
```

### 方法二：在代码中调用

```python
from openalex_scholar import OpenAlexScholar

scholar = OpenAlexScholar(email="your@email.com")
papers = scholar.search_papers("linear programming optimization")

for paper in papers:
    print(f"标题: {paper.title}")
    print(f"作者: {', '.join(paper.authors)}")
    print(f"年份: {paper.publication_year}")
    print(f"引用: {paper.citation_format}")
    print(f"DOI: {paper.doi}")
    print("---")
```

### 方法三：获取字典格式数据

```python
from openalex_scholar import OpenAlexScholar

scholar = OpenAlexScholar(email="your@email.com")
papers = scholar.search_papers("linear programming optimization")

for paper in papers:
    paper_dict = paper.to_dict()
    print(f"标题: {paper_dict['title']}")
    print(f"作者: {paper_dict['authors']}")
    print(f"年份: {paper_dict['publication_year']}")
    print(f"引用: {paper_dict['citation_format']}")
    print("---")
```

---

## 返回字段说明

### Paper 对象属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `title` | str | 论文标题 |
| `abstract` | str/None | 摘要（从倒排索引重建） |
| `authors` | List[str] | 作者列表 |
| `cited_by_count` | int | 被引用次数 |
| `doi` | str/None | DOI 标识符 |
| `publication_year` | int/None | 发表年份 |

### Paper 对象方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `citation_format` | str | 格式化引用文本（APA风格） |
| `to_dict()` | Dict | 转换为字典格式，包含所有字段和citation_format |

---

## 引用格式示例

```
Smith J. et al. (2020). A Novel Approach to Grey Prediction Models. 
DOI: 10.1016/j.example.2020.123456
```

---

## API 限制与最佳实践

| 项目 | 说明 |
|------|------|
| **速率限制** | 使用礼貌池(polite pool)可提高限制 |
| **必需参数** | 必须提供 `mailto` 邮箱参数 |
| **返回数量** | 默认返回 8 篇相关论文 |
| **搜索字段** | 支持标题、摘要、作者等多字段搜索 |

---

## 数学建模常用搜索关键词

### 优化算法
- `linear programming optimization`
- `genetic algorithm optimization`
- `particle swarm optimization`
- `simulated annealing`

### 预测模型
- `grey prediction model GM(1,1)`
- `ARIMA time series forecasting`
- `LSTM neural network prediction`
- `prophet forecasting`

### 评价方法
- `analytic hierarchy process AHP`
- `TOPSIS multi-criteria decision`
- `entropy weight method`
- `data envelopment analysis DEA`

### 图论与网络
- `shortest path algorithm Dijkstra`
- `minimum spanning tree`
- `maximum flow network`
- `vehicle routing problem`

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/openalex_scholar.py` | OpenAlex 搜索实现脚本 |

---

## 参考链接

- [OpenAlex API 文档](https://docs.openalex.org/)
- [OpenAlex 礼貌池规则](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication)
