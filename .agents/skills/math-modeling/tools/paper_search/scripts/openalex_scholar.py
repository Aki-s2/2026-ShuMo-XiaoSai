#!/usr/bin/env python3
"""
OpenAlex Scholar - 学术论文搜索工具

通过 OpenAlex API 搜索学术论文，为数学建模提供参考文献支持。
"""

import argparse
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


__all__ = ['Paper', 'OpenAlexScholar']


@dataclass
class Paper:
    """论文数据类"""
    title: str
    authors: List[str]
    publication_year: Optional[int]
    cited_by_count: int
    doi: Optional[str]
    abstract: Optional[str]
    
    @property
    def citation_format(self) -> str:
        """生成引用格式"""
        author_str = ", ".join(self.authors[:3]) if self.authors else "Unknown"
        if len(self.authors) > 3:
            author_str += " et al."
        year_str = f" ({self.publication_year})" if self.publication_year else ""
        doi_str = f" DOI: {self.doi}" if self.doi else ""
        return f"{author_str}{year_str}. {self.title}.{doi_str}"
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'title': self.title,
            'authors': self.authors,
            'publication_year': self.publication_year,
            'cited_by_count': self.cited_by_count,
            'doi': self.doi,
            'abstract': self.abstract,
            'citation_format': self.citation_format
        }


class OpenAlexScholar:
    """OpenAlex 学术搜索类"""
    
    def __init__(self, email: str = None):
        """
        初始化搜索器
        
        Args:
            email: 用于礼貌池的邮箱地址
        """
        self.base_url = "https://api.openalex.org/works"
        self.email = email
    
    def search_papers(self, query: str, limit: int = 8) -> List[Paper]:
        """
        搜索论文
        
        Args:
            query: 搜索关键词
            limit: 返回结果数量
            
        Returns:
            论文列表
        """
        params = {
            "search": query,
            "per_page": limit,
            "select": "id,title,display_name,authorships,cited_by_count,doi,publication_year,biblio,abstract_inverted_index",
        }
        
        if self.email:
            params["mailto"] = self.email
        
        query_string = urllib.parse.urlencode(params)
        url = f"{self.base_url}?{query_string}"
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": f"OpenAlexScholar (mailto:{self.email})" if self.email else "OpenAlexScholar"
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                return self._parse_results(data)
                
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def _parse_results(self, data: Dict) -> List[Paper]:
        """解析API返回结果"""
        papers = []
        
        for work in data.get("results", []):
            # 提取作者信息
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                author_name = author.get("display_name", "")
                if author_name:
                    authors.append(author_name)
            
            # 提取摘要
            abstract = None
            abstract_index = work.get("abstract_inverted_index")
            if abstract_index:
                abstract = self._get_abstract_from_index(abstract_index)
            
            paper = Paper(
                title=work.get("display_name", "Unknown Title"),
                authors=authors,
                publication_year=work.get("publication_year"),
                cited_by_count=work.get("cited_by_count", 0),
                doi=work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
                abstract=abstract
            )
            papers.append(paper)
        
        return papers
    
    def _get_abstract_from_index(self, abstract_inverted_index: Dict) -> str:
        """从倒排索引重建摘要"""
        try:
            max_position = max(max(positions) for positions in abstract_inverted_index.values())
            words = [""] * (max_position + 1)
            
            for word, positions in abstract_inverted_index.items():
                for position in positions:
                    words[position] = word
            
            return " ".join(words).strip()
        except Exception:
            return ""


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="OpenAlex 学术论文搜索工具")
    parser.add_argument("--query", "-q", required=True, help="搜索关键词")
    parser.add_argument("--email", "-e", default="example@example.com", help="邮箱地址（用于礼貌池）")
    parser.add_argument("--limit", "-n", type=int, default=8, help="返回结果数量（默认8）")
    parser.add_argument("--json", "-j", action="store_true", help="以JSON格式输出")
    
    args = parser.parse_args()
    
    print(f"正在搜索: {args.query}")
    print(f"邮箱: {args.email}")
    print("-" * 80)
    
    scholar = OpenAlexScholar(email=args.email)
    papers = scholar.search_papers(args.query, limit=args.limit)
    
    if not papers:
        print("未找到相关论文")
        return
    
    print(f"找到 {len(papers)} 篇相关论文:\n")
    
    for i, paper in enumerate(papers, 1):
        if args.json:
            print(json.dumps({
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.publication_year,
                "citations": paper.cited_by_count,
                "doi": paper.doi,
                "abstract": paper.abstract[:200] + "..." if paper.abstract and len(paper.abstract) > 200 else paper.abstract
            }, ensure_ascii=False, indent=2))
        else:
            print(f"[{i}] {paper.title}")
            print(f"    作者: {', '.join(paper.authors[:5])}{' et al.' if len(paper.authors) > 5 else ''}")
            print(f"    年份: {paper.publication_year or 'Unknown'}")
            print(f"    引用: {paper.cited_by_count}")
            if paper.doi:
                print(f"    DOI: {paper.doi}")
            if paper.abstract:
                abstract_preview = paper.abstract[:150] + "..." if len(paper.abstract) > 150 else paper.abstract
                print(f"    摘要: {abstract_preview}")
            print()


if __name__ == "__main__":
    main()
