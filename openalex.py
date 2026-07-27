# ==================================================================
# openalex.py -- APOLLO v10.0.0 OpenALEX API通信モジュール
# OpenALEX Collector 互換の検索・取得ロジック（Python 実装）
# ==================================================================

import re
import html
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

    def __init__(self, api_key: str = None):
        # OpenAlex API は 2026-02-13 より API キーが必須（従来の mailto / polite pool 方式は廃止）。
        # キーは UI（各ユーザーが自分のキーを入力）からのみ受け取る。サーバ既定キー
        # （環境変数フォールバック）は意図的に持たない＝公開時に全訪問者が運用者キーを
        # 共有してしまうのを防ぐ。アプリ側でキー入力を必須とし（未入力では検索不可）、
        # キーなしのお試し枠運用は行わない。
        self.api_key = (api_key or "").strip()

    def _auth_params(self) -> dict:
        """認証用の共通パラメータ。api_key を優先し、後方互換のため mailto も併記する。"""
        params = {"mailto": self.MAILTO}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _redact(self, text) -> str:
        """エラーメッセージ等から API キーを伏せる。

        requests の例外メッセージには失敗 URL（`...?api_key=<キー>`）が平文で含まれるため、
        UI 表示やサーバログに API キーが漏れないようマスクする（公開 Web アプリ対策）。
        """
        s = str(text)
        # クエリ文字列の api_key=... を伏せる（URL がキー値と異なる表現でも確実に消す）
        s = re.sub(r"(api_key=)[^&\s'\"]+", r"\1***", s, flags=re.IGNORECASE)
        # 念のためキー実値そのものも伏せる
        if self.api_key:
            s = s.replace(self.api_key, "***")
        return s

    def test_connection(self) -> tuple:
        """API キーの有効性を確認する。(成功フラグ, メッセージ) を返す。"""
        try:
            params = {"per_page": 1, **self._auth_params()}
            resp = requests.get(self.BASE_URL, params=params, timeout=30)
            if resp.status_code == 200:
                if self.api_key:
                    return True, "接続成功（API キーは有効です）"
                return False, "API キーが未設定です。本アプリは API キーが必須です。"
            if resp.status_code in (401, 403):
                return False, f"認証エラー（{resp.status_code}）: API キーが無効か、上限を超過しています。"
            return False, f"接続失敗（HTTP {resp.status_code}）"
        except Exception as exc:
            return False, self._redact(f"接続エラー: {exc}")

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
            **self._auth_params(),
        }
        resp = requests.get(self.INST_URL, params=params, timeout=30)
        try:
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            # 例外メッセージに失敗 URL（api_key を含む）が載るため伏せて再送出
            raise RuntimeError(self._redact(f"機関検索エラー: {exc}")) from None
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
            **self._auth_params(),
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

        # フィールド限定検索（コマンドラインモード）: top-level search= の代わりに
        # <field>.search フィルタを使う。query 中のカンマは filter 区切りと衝突するため空白化。
        scope_key = {
            "title": "title.search",
            "abstract": "abstract.search",
            "title_and_abstract": "title_and_abstract.search",
            "fulltext": "fulltext.search",
        }.get(params.get("search_scope"))
        if scope_key and query:
            filters.append(f"{scope_key}:{query.replace(',', ' ')}")
            url_params.pop("search", None)

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

                # 429 以外の 4xx（不正なクエリ・無効なキー等）はリトライしても直らない。
                # 5 回の無駄な待機を避け、OpenAlex のエラーメッセージを添えて即時に失敗させる。
                if 400 <= resp.status_code < 500:
                    detail = ""
                    try:
                        _j = resp.json()
                        detail = _j.get("message") or _j.get("error") or ""
                    except Exception:
                        detail = (resp.text or "")[:300]
                    # detail は OpenAlex 由来でキーを含まないが、念のため伏せる
                    raise RuntimeError(
                        self._redact(f"APIリクエストエラー（HTTP {resp.status_code}）: {detail}".strip())
                    ) from None

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
                # 例外メッセージに失敗 URL（api_key を含む）が載るため伏せて再送出
                # （from None で原例外チェーンも切り、トレースバック経由の漏洩も防ぐ）
                raise RuntimeError(self._redact(f"API通信エラー: {exc}")) from None

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
        search_scope: str = None,
        on_progress: Callable = None,
    ) -> list[dict]:
        """単一クエリで論文を検索する（cursor-based pagination）

        search_scope を指定すると、top-level の search= ではなく
        title.search / abstract.search / title_and_abstract.search / fulltext.search
        フィルタで検索する（コマンドラインモードのフィールド限定用）。

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
            "search_scope": search_scope,
        }

        all_papers: list[dict] = []
        seen_ids: set = set()
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

            # 重複除去（cursor の揺らぎやリトライ境界での二重取得を防ぐ。search_by_year / search_multi_query と整合）
            for _p in page["results"]:
                _pid = _p.get("id")
                if _pid and _pid in seen_ids:
                    continue
                if _pid:
                    seen_ids.add(_pid)
                all_papers.append(_p)

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
                # コマンドライン検索式の候補取得で <field>.search を年別にも効かせる
                "search_scope": kwargs.get("search_scope"),
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
    # Command-line search (TI=/AB=/TA=/TX=/FT= + AND/OR/NOT + near/adj + wildcard)
    # ------------------------------------------------------------------
    def search_command_query(
        self,
        raw_query: str,
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
        by_year: bool = False,
        max_per_year: int = 10000,
    ) -> list[dict]:
        """コマンドライン検索式で論文を取得する（OpenALEX Collector 互換）。

        2フェーズ方式:
          ① 検索式を AST に解析し、フィールドに応じた scope（title/abstract/title_and_abstract/
             all/fulltext）で OpenAlex 候補クエリを構築・取得（OR は代替案に展開、NOT は除外して取得）。
          ② near/adj・NOT・ワイルドカードを含む場合は、取得した候補をローカルで厳密照合して絞り込む。

        by_year=True のときは、各候補クエリを年別取得（search_by_year）して 10,000 件/クエリ
        制限を回避する。この場合 max_results は無視し、年範囲全体で一致した論文を全件返す
        （通常モードの search_by_year と整合）。on_progress は
        (cand_index, n_cand, year_index, total_years, year, year_count, year_total, all_count) で呼ぶ。

        by_year=False のとき on_progress は (current, total) で呼ぶ（従来通り）。

        Returns: OpenAlex raw 論文 dict のリスト（search() と同形式）。
        構文エラー時は openalex_query.QueryError を送出する。
        """
        import openalex_query as oq

        plan = oq.compile_command_query(raw_query)   # QueryError を送出しうる
        scope = plan["scope"]
        cand_queries = plan["candidate_queries"]
        needs_filter = plan["needs_local_filter"]
        ast = plan["ast"]
        default_field = plan["default_field"]

        def _apply_local_filter(plist: list[dict]) -> list[dict]:
            """② near/adj・NOT・ワイルドカードのローカル厳密照合（必要時のみ）。"""
            if not needs_filter:
                return plist
            out = []
            for p in plist:
                title = self._strip_tags(p.get("title", "") or "")
                abstract = self.reconstruct_abstract(p.get("abstract_inverted_index"))
                if oq.paper_matches(ast, title, abstract, default_field):
                    out.append(p)
            return out

        # 各候補クエリに渡す共通フィルタ引数
        _kw = dict(
            sort=sort, min_citations=min_citations, pub_types=pub_types,
            oa_only=oa_only, institution_ids=institution_ids,
            has_abstract=has_abstract, language=language,
        )

        seen = set()
        papers: list[dict] = []
        cand_list = [cq for cq in cand_queries if cq]
        n_cand = max(1, len(cand_list))

        # --- 年別取得モード（候補クエリごとに年別取得し、10,000 件/クエリ制限を回避） ---
        if by_year:
            for ci, cq in enumerate(cand_list):
                def _year_cb(yi, total_years, year, year_count, year_total, cand_count,
                             _ci=ci, _nc=n_cand):
                    if on_progress:
                        # 統合累計 = 確定済み候補ぶん(len(papers)) + 当候補の実時間取得数(cand_count)
                        on_progress(_ci, _nc, yi, total_years, year,
                                    year_count, year_total, len(papers) + cand_count)
                batch = self.search_by_year(
                    cq, year_from, year_to, max_per_year=max_per_year,
                    search_scope=scope, on_progress=_year_cb, **_kw,
                )
                for p in batch:
                    pid = p.get("id")
                    if pid and pid in seen:
                        continue
                    if pid:
                        seen.add(pid)
                    papers.append(p)
            return _apply_local_filter(papers)

        # --- 通常モード（年範囲を合算して取得上限まで） ---
        # ローカル絞り込みがある場合は取り溢れ対策で多めに集める（上限は API 負荷のため抑制）
        accum_cap = max_results * 4 if needs_filter else max_results
        for cq in cand_list:
            results = self.search(
                cq, year_from=year_from, year_to=year_to,
                max_results=max_results, search_scope=scope, **_kw,
            )
            for p in results:
                pid = p.get("id")
                if pid and pid in seen:
                    continue
                if pid:
                    seen.add(pid)
                papers.append(p)
            if on_progress:
                on_progress(min(len(papers), accum_cap), accum_cap)
            if len(papers) >= accum_cap:
                break

        return _apply_local_filter(papers)[:max_results]

    # ------------------------------------------------------------------
    # Abstract reconstruction
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_tags(s) -> str:
        """要旨・タイトルに残る HTML/JATS タグの残骸と文字実体参照を除去し、空白を正規化する。

        OpenAlex の abstract_inverted_index やタイトルには <i> / <sub> / <jats:p> 等のタグや
        &amp; 等の実体参照が混入することがあるため、SBERT 埋め込み前にクリーニングする
        （参考: OpenALEX Collector の _stripTags 相当）。
        """
        if s is None or s == "":
            return ""
        s = str(s)
        # 1) 文字実体参照を先に復元する（&lt;p&gt; → <p>、&#x0D;/&#13; → CR、&amp; → & 等）。
        #    順序が重要: 先にタグ除去すると、エスケープされたタグ（&lt;p class="..."&gt; 等）が
        #    後段の復元で <p ...> として復活し残ってしまう。html.unescape は名前付き・10進・16進
        #    （&#x0D; 等）の全実体を一括復元する。
        s = html.unescape(s)
        # 2) HTML/JATS タグを除去（<p>, </p>, <p class="E-JOURNAL...">, <jats:p>, <sub>, <i> 等）。
        #    数式の "a < b" を誤除去しないよう、開始は英字またはスラッシュに限定する。
        s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)
        # 3) 復元された制御文字（CR/LF/タブ・&#x0D; 由来の \r 等）と連続空白を 1 つの空白へ正規化
        s = re.sub(r"\s+", " ", s).strip()
        return s

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
        return OpenAlexCollector._strip_tags(" ".join(p[1] for p in pairs))

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
            "title": self._strip_tags(raw.get("title", "") or ""),
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

        線形回帰の傾きでトレンドを判定し、複利成長率 (CAGR) を算出する。

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
