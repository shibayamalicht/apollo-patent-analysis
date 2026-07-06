# -*- coding: utf-8 -*-
"""キーワード抽出のプロセス並列用 軽量ワーカーモジュール。

streamlit など重いものを import しないことで、loky/multiprocessing のワーカー起動を軽くし、
Streamlit ランタイム下でのサブプロセス起動を安全にする（patiroha のみ使用）。
utils.extract_keywords_batch から参照される。
"""


def extract_keywords_worker(text, stopwords, clean_html=False):
    """1テキストから複合名詞キーワードを抽出する（picklable・streamlit非依存）。

    utils.extract_keywords と同じサニタイズ（非文字列/空/8000字超の切り詰め）を行い、
    例外は空リストで吸収する。
    """
    try:
        import patiroha
        if not isinstance(text, str):
            return []
        text = text.strip()
        if not text:
            return []
        if len(text) > 8000:
            text = text[:8000]
        return patiroha.extract_keywords(text, stopwords=stopwords, clean_html=clean_html)
    except Exception:
        return []
