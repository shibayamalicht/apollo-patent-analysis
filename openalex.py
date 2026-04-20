# ==================================================================
# openalex.py -- APOLLO v8.0.0 OpenALEX API通信モジュール
# OpenALEX_Collector.html のJSロジックをPython移植
# ==================================================================

import time
import math
import requests
import pandas as pd
from typing import Optional, Callable


class OpenAlexCollector:
    """OpenAlex APIを使った学術論文収集クラス"""

    BASE_URL = "https://api.openalex.org/works"
    INST_URL = "https://api.openalex.org/institutions"
    RATE_LIMIT = 0.15          # 150ms between requests
    MAILTO = "apollo-v7@example.com"  # polite pool
    MAX_RETRIES = 5
    PER_PAGE = 200             # API最大値

    # ------------------------------------------------------------------
    # Institution resolution
    # ------------------------------------------------------------------
    def resolve_institution(self, name: str) -> Optional[dict]:
        """機関名からOpenAlex Institution IDを解決する

        Parameters
        ----------
        name : str
            検索する機関名（部分一致可）

        Returns
        -------
        dict or None
            {"id": "https://openalex.org/I...", "display_name": "..."}
            見つからなければ None
        """
        params = {
            "search": name.strip(),
            "per_page": 5,
            "mailto": self.MAILTO,
        }
        resp = requests.get(self.INST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        return {
            "id": results[0]["id"],
            "display_name": results[0]["display_name"],
        }

    # ------------------------------------------------------------------
    # Low-level single-page fetch (with retry / rate-limit)
    # ------------------------------------------------------------------
    def _fetch_page(self, query: str, params: dict, cursor: str = "*") -> dict:
        """1ページ分のAPI呼び出し（リトライ・レート制限付き）

        Returns
        -------
        dict  {"results": [...], "total": int, "next_cursor": str|None}
        """
        url_params = {
            "search": query,
            "per_page": min(params.get("per_page", self.PER_PAGE), self.PER_PAGE),
            "cursor": cursor,
            "mailto": self.MAILTO,
        }

        # sort
        sort = params.get("sort")
        if sort and sort != "relevance":
            url_params["sort"] = sort

        # filters
        filters: list[str] = []
        year_from = params.get("year_from")
        year_to = params.get("year_to")
        if year_from and year_to:
            filters.append(f"publication_year:{year_from}-{year_to}")
        elif year_from:
            filters.append(f"publication_year:{year_from}-")
        elif year_to:
            filters.append(f"publication_year:-{year_to}")

        min_citations = params.get("min_citations", 0)
        if min_citations and min_citations > 0:
            filters.append(f"cited_by_count:>{min_citations - 1}")

        pub_types = params.get("pub_types")
        if pub_types:
            filters.append(f"type:{('|').join(pub_types)}")

        if params.get("oa_only"):
            filters.append("is_oa:true")

        # 要約ありのみ（分析精度確保用）
        if params.get("has_abstract"):
            filters.append("has_abstract:true")

        # 言語フィルタ（例: "en" で英語のみ）
        language = params.get("language")
        if language:
            filters.append(f"language:{language}")

        institution_ids = params.get("institution_ids")
        if institution_ids:
            filters.append(
                f"authorships.institutions.lineage:{('|').join(institution_ids)}"
            )

        if filters:
            url_params["filter"] = ",".join(filters)

        # retry loop
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(self.BASE_URL, params=url_params, timeout=60)

                if resp.status_code == 429:
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(5.0 * (attempt + 1))
                        continue
                    raise RuntimeError(
                        "レート制限です。少し待ってから再試行してください。"
                    )

                resp.raise_for_status()
                data = resp.json()
                meta = data.get("meta", {})
                return {
                    "results": data.get("results", []),
                    "total": meta.get("count", 0),
                    "next_cursor": meta.get("next_cursor"),
                }

            except requests.exceptions.RequestException as exc:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"API通信エラー: {exc}") from exc

        # should not reach here
        raise RuntimeError("API通信に失敗しました（最大リトライ回数超過）")

    # ------------------------------------------------------------------
    # search: single query, cursor-based pagination
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        year_from: int = None,
        year_to: int = None,
        max_results: int = 200,
        sort: str = "relevance",
        min_citations: int = 0,
        pub_types: list = None,
        oa_only: bool = False,
        institution_ids: list = None,
        has_abstract: bool = False,
        language: str = None,
        on_progress: Callable = None,
    ) -> list[dict]:
        """単一クエリで論文を検索する（cursor-based pagination）

        Parameters
        ----------
        query : str
            検索キーワード
        year_from, year_to : int, optional
            出版年フィルタ
        max_results : int
            最大取得件数（デフォルト 200）
        sort : str
            ソート順 ("relevance", "cited_by_count:desc", "publication_date:desc", ...)
        min_citations : int
            最小引用数フィルタ
        pub_types : list[str], optional
            論文種別フィルタ ("article", "review", ...)
        oa_only : bool
            オープンアクセスのみ
        institution_ids : list[str], optional
            機関ID（OpenAlex形式）のリスト
        on_progress : callable, optional
            (current_count, total_count) を受け取るコールバック

        Returns
        -------
        list[dict]  API raw results
        """
        params = {
            "year_from": year_from,
            "year_to": year_to,
            "sort": sort,
            "min_citations": min_citations,
            "pub_types": pub_types,
            "oa_only": oa_only,
            "institution_ids": institution_ids,
            "has_abstract": has_abstract,
            "language": language,
        }

        all_papers: list[dict] = []
        cursor = "*"
        total = None

        while len(all_papers) < max_results:
            remaining = max_results - len(all_papers)
            params["per_page"] = min(self.PER_PAGE, remaining)

            page = self._fetch_page(query, params, cursor)

            if total is None:
                total = min(page["total"], max_results)

            if not page["results"]:
                break

            all_papers.extend(page["results"])

            if on_progress:
                on_progress(len(all_papers), total)

            if not page["next_cursor"] or len(all_papers) >= max_results:
                break

            cursor = page["next_cursor"]
            time.sleep(self.RATE_LIMIT)

        return all_papers[:max_results]

    # ------------------------------------------------------------------
    # search_by_year: year-by-year to bypass 10k limit
    # ------------------------------------------------------------------
    def search_by_year(
        self,
        query: str,
        year_from: int,
        year_to: int,
        max_per_year: int = 10000,
        on_progress: Callable = None,
        **kwargs,
    ) -> list[dict]:
        """年別に検索して10,000件制限を回避する

        Parameters
        ----------
        query : str
        year_from, year_to : int
        max_per_year : int
            年あたりの最大取得件数
        on_progress : callable, optional
            (year_index, total_years, year, year_count, year_total, all_count)
        **kwargs
            search() に渡す追加パラメータ

        Returns
        -------
        list[dict]  重複除去済み raw results
        """
        years = list(range(year_from, year_to + 1))
        all_papers: list[dict] = []
        seen_ids: set[str] = set()

        for yi, year in enumerate(years):
            cursor = "*"
            year_count = 0
            year_total = None

            params = {
                "year_from": year,
                "year_to": year,
                "sort": kwargs.get("sort", "relevance"),
                "min_citations": kwargs.get("min_citations", 0),
                "pub_types": kwargs.get("pub_types"),
                "oa_only": kwargs.get("oa_only", False),
                "institution_ids": kwargs.get("institution_ids"),
                "has_abstract": kwargs.get("has_abstract", False),
                "language": kwargs.get("language"),
            }

            while year_count < max_per_year:
                remaining = max_per_year - year_count
                params["per_page"] = min(self.PER_PAGE, remaining)

                page = self._fetch_page(query, params, cursor)

                if year_total is None:
                    year_total = min(page["total"], max_per_year)

                if not page["results"]:
                    break

                for paper in page["results"]:
                    pid = paper.get("id", "")
                    if pid and pid not in seen_ids:
                        seen_ids.add(pid)
                        all_papers.append(paper)

                year_count += len(page["results"])

                if on_progress:
                    on_progress(
                        yi, len(years), year,
                        year_count, year_total or 0,
                        len(all_papers),
                    )

                if not page["next_cursor"] or year_count >= max_per_year:
                    break

                cursor = page["next_cursor"]
                time.sleep(self.RATE_LIMIT)

        return all_papers

    # ------------------------------------------------------------------
    # search_multi_query: OR across multiple queries
    # ------------------------------------------------------------------
    def search_multi_query(
        self,
        queries: list[str],
        max_results: int = 200,
        on_progress: Callable = None,
        **kwargs,
    ) -> list[dict]:
        """複数クエリをOR統合（各クエリを個別検索、重複除去）

        Parameters
        ----------
        queries : list[str]
        max_results : int
            クエリあたりの最大件数
        on_progress : callable, optional
            (query_index, total_queries, current, total)
        **kwargs
            search() に渡す追加パラメータ

        Returns
        -------
        list[dict]  重複除去済み raw results
        """
        all_papers: list[dict] = []
        seen_ids: set[str] = set()

        for qi, query in enumerate(queries):
            query = query.strip()
            if not query:
                continue

            def batch_progress(current, total):
                if on_progress:
                    on_progress(qi, len(queries), current, total)

            batch = self.search(
                query,
                max_results=max_results,
                on_progress=batch_progress,
                **kwargs,
            )

            for paper in batch:
                pid = paper.get("id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_papers.append(paper)

            if qi < len(queries) - 1:
                time.sleep(self.RATE_LIMIT)

        return all_papers

    # ------------------------------------------------------------------
    # Abstract reconstruction
    # ------------------------------------------------------------------
    @staticmethod
    def reconstruct_abstract(inverted_index: dict) -> str:
        """OpenAlexのabstract_inverted_indexから原文を復元する

        Parameters
        ----------
        inverted_index : dict
            {word: [pos1, pos2, ...]} 形式の逆インデックス

        Returns
        -------
        str  復元されたアブストラクト文字列
        """
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        pairs: list[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                pairs.append((pos, word))
        pairs.sort(key=lambda p: p[0])
        return " ".join(p[1] for p in pairs)

    # ------------------------------------------------------------------
    # Paper transformation
    # ------------------------------------------------------------------
    def transform_paper(self, raw: dict) -> dict:
        """OpenAlex APIのraw応答を統一形式に変換する

        Parameters
        ----------
        raw : dict
            OpenAlex APIの1論文レスポンス

        Returns
        -------
        dict  統一形式:
            paper_id, title, abstract, year, authors, institutions,
            venue, citation_count, publication_date, doi
        """
        authorships = raw.get("authorships", [])

        # authors
        authors = "; ".join(
            a.get("author", {}).get("display_name", "")
            for a in authorships
            if a.get("author", {}).get("display_name")
        )

        # doi
        doi = raw.get("doi", "") or ""
        if doi:
            doi = doi.replace("https://doi.org/", "")

        # venue
        primary = raw.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = source.get("display_name", "")

        # paper_id
        oa_id = raw.get("id", "")
        paper_id = oa_id.replace("https://openalex.org/", "")

        # abstract
        abstract = self.reconstruct_abstract(
            raw.get("abstract_inverted_index")
        )

        # institutions (deduplicated)
        inst_set: set[str] = set()
        for a in authorships:
            for inst in a.get("institutions") or []:
                name = inst.get("display_name", "")
                if name:
                    inst_set.add(name)
        institutions = "; ".join(sorted(inst_set))

        return {
            "paper_id": paper_id,
            "title": raw.get("title", "") or "",
            "abstract": abstract,
            "year": raw.get("publication_year"),
            "authors": authors,
            "institutions": institutions,
            "venue": venue,
            "citation_count": raw.get("cited_by_count", 0) or 0,
            "publication_date": raw.get("publication_date", "") or "",
            "doi": doi,
        }

    # ------------------------------------------------------------------
    # NPL DataFrame conversion
    # ------------------------------------------------------------------
    def to_npl_dataframe(self, papers: list[dict]) -> pd.DataFrame:
        """変換済み論文リストをNPL統一形式のDataFrameに変換する

        Parameters
        ----------
        papers : list[dict]
            transform_paper() で変換済みの論文辞書リスト

        Returns
        -------
        pd.DataFrame
            カラム: unified_title, unified_content, unified_date,
                    unified_source, unified_region, unified_status,
                    data_sub_type, source_filename,
                    citation_count, doi
        """
        rows = []
        for p in papers:
            rows.append(
                {
                    "unified_title": p.get("title", ""),
                    "unified_content": p.get("abstract", ""),
                    "unified_date": p.get("publication_date", ""),
                    "unified_source": p.get("venue", ""),
                    "unified_region": p.get("institutions", ""),
                    "unified_status": "",
                    "data_sub_type": "Academic",
                    "source_filename": f"OpenAlex_{p.get('paper_id', '')}",
                    "citation_count": p.get("citation_count", 0),
                    "doi": p.get("doi", ""),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # CAGR / trend calculation
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_cagr(papers: list[dict]) -> dict:
        """年別件数からCAGRとトレンドを計算する

        OpenALEX_Collector.html の SearchController.calculateCAGR と
        同一ロジック。線形回帰の傾きでトレンドを判定し、複利成長率
        (CAGR) を算出する。

        Parameters
        ----------
        papers : list[dict]
            transform_paper() 済みの論文リスト（year キー必須）

        Returns
        -------
        dict  {
            "cagr": float or None,
            "trend": str,       # "急上昇" / "増加傾向" / "横ばい" / "減少傾向" / "急減少" / "不明"
            "slope": float or None,
            "year_counts": dict  # {year: count}
        }
        """
        # 年別集計
        year_counts: dict[int, int] = {}
        for p in papers:
            y = p.get("year")
            if y is not None:
                year_counts[y] = year_counts.get(y, 0) + 1

        years = sorted(year_counts.keys())
        if len(years) < 2:
            return {
                "cagr": None,
                "trend": "不明",
                "slope": None,
                "year_counts": year_counts,
            }

        counts = [year_counts[y] for y in years]
        n = len(years)

        # 線形回帰
        sum_x = sum(years)
        sum_y = sum(counts)
        sum_xy = sum(x * c for x, c in zip(years, counts))
        sum_x2 = sum(x * x for x in years)

        denominator = n * sum_x2 - sum_x * sum_x
        slope = (n * sum_xy - sum_x * sum_y) / denominator if denominator else 0.0

        # トレンド判定
        avg_count = sum_y / n
        normalized_slope = slope / avg_count if avg_count > 0 else 0.0

        if normalized_slope > 0.15:
            trend = "急上昇"
        elif normalized_slope > 0.03:
            trend = "増加傾向"
        elif normalized_slope > -0.03:
            trend = "横ばい"
        elif normalized_slope > -0.15:
            trend = "減少傾向"
        else:
            trend = "急減少"

        # CAGR
        first_nonzero = next((c for c in counts if c > 0), None)
        last_val = counts[-1]
        n_years = years[-1] - years[0]

        cagr = None
        if n_years > 0 and first_nonzero and first_nonzero > 0 and last_val > 0:
            cagr = math.pow(last_val / first_nonzero, 1.0 / n_years) - 1.0

        return {
            "cagr": cagr,
            "trend": trend,
            "slope": slope,
            "year_counts": year_counts,
        }
