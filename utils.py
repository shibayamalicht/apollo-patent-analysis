import streamlit as st
import platform
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import re
import json
import pandas as pd
import numpy as np
import patiroha
import plotly.express as px

# ==================================================================
# --- 0. APOLLO Standard 色定数 ---
# ==================================================================
APOLLO_COLORS = px.colors.qualitative.G10      # Plotly 離散カラーパレット (10色)
APOLLO_BG = "#ffffff"                          # 背景色 (paper_bgcolor / plot_bgcolor)
APOLLO_TEXT = "#333333"                        # 文字色 (font.color)
APOLLO_ACCENT = "#003366"                      # アクセント色 (H1タイトル等)
APOLLO_TEMPLATE = "plotly_white"               # Plotly テンプレート

# ==================================================================
# --- 1. フォント設定 (共通) ---
# ==================================================================
def get_japanese_font_path():
    """OSを判定して適切な日本語フォントパスを返す"""
    system = platform.system()
    font_paths = []
    
    if system == "Darwin": # Mac
        font_paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans W3.ttc",
            "/System/Library/Fonts/Hiragino Kaku Gothic ProN.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc" 
        ]
    elif system == "Windows": # Windows
        font_paths = [
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/yugothr.ttc",
            "C:/Windows/Fonts/YuGothR.ttc"
        ]
    else: # Linux (Streamlit Cloudなど)
        font_paths = [
            "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto/NotoSansCJKjp-Regular.otf"
        ]
        
    for path in font_paths:
        if os.path.exists(path): return path
    return None

def configure_matplotlib_font():
    """Matplotlibのフォント設定を適用する"""
    font_path = get_japanese_font_path()
    if font_path:
        try:
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
            return font_path
        except:
            pass
    return None

# ==================================================================
# --- 2. ストップワード (共通) — patiroha委譲 ---
# ==================================================================

def get_stopwords(mode='patent'):
    """ストップワード取得 (patiroha委譲)"""
    return patiroha.get_stopwords(mode)

def get_patent_stopwords():
    """特許分析用のストップワード"""
    return patiroha.get_stopwords("patent")

def get_npl_stopwords():
    """NPL分析用のストップワード"""
    return patiroha.get_stopwords("npl")

# ==================================================================
# --- 3. サイドバー設定 (共通) ---
# ==================================================================
def render_sidebar():
    """共通サイドバーを描画する"""


    # 共通CSSの適用
    st.markdown("""
    <style>
        html, body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #ffffff; color: #333333; }
        [data-testid="stSidebar"] { background-color: #f8f9fa; }
        [data-testid="stHeader"] { background-color: #ffffff; }
        
        /* H1 Title Spacing */
        [data-testid="stSidebar"] h1 { 
            color: #003366; 
            font-weight: 900 !important; 
            font-size: 2.5rem !important; 
            margin-top: 0 !important; 
            padding-top: 0 !important; 
            margin-bottom: 0 !important;
        }
        h1 { color: #003366; font-weight: 700; }
        h2, h3 { color: #333333; font-weight: 500; border-bottom: 2px solid #f0f0f0; padding-bottom: 5px; }
        
        /* Hide default nav */
        [data-testid="stSidebarNav"] { display: none !important; }
        
        /* Remove Top Whitespace (Robust Selectors) */
        section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem; }
        [data-testid="stSidebarUserContent"] { padding-top: 0rem; }
        [data-testid="stSidebar"] .block-container { padding-top: 0rem; padding-bottom: 1rem; }
        
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton>button { font-weight: 600; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background-color: #f0f2f6; border-radius: 8px 8px 0 0; padding: 10px 15px; }
        .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #003366; }
        [data-testid="stSidebar"] h3 { border-bottom: none !important; padding-bottom: 10px !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("APOLLO")
        st.markdown("""
Advanced Patent & Overall Landscape-analytics
Logic Orbiter

**v7.0.0**
""")
        st.markdown("---")
        st.subheader("Home"); st.page_link("Home.py", label="Mission Control", icon="🛰️")
        st.subheader("Modules")
        st.page_link("pages/1_🌍_ATLAS.py", label="ATLAS", icon="🌍")
        st.page_link("pages/2_💡_CORE.py", label="CORE", icon="💡")
        st.page_link("pages/3_🚀_Saturn_V.py", label="Saturn V", icon="🚀")
        st.page_link("pages/7_🦅_EAGLE.py", label="EAGLE", icon="🦅")
        st.page_link("pages/4_📈_MEGA.py", label="MEGA", icon="📈")
        st.page_link("pages/5_🧭_Explorer.py", label="Explorer", icon="🧭")
        st.page_link("pages/6_🔗_CREW.py", label="CREW", icon="🔗")
        st.page_link("pages/9_🌌_NEBULA.py", label="NEBULA", icon="🌌")
        st.page_link("pages/8_📝_VOYAGER.py", label="VOYAGER", icon="📝")
        st.page_link("pages/10_📡_CAPCOM.py", label="CAPCOM", icon="📡")
        st.markdown("---")

        # --- CAPCOM ステータスインジケーター ---
        try:
            import capcom
            if capcom.is_active():
                _sid = st.session_state.get('capcom_session_id', '')
                # session_stateベースのテレメトリ（ファイルI/Oなし）
                _tel = capcom.get_telemetry()
                _sc, _pc, _dc = _tel['snapshots'], _tel['prompts'], _tel['data']
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 100%);
                            border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
                            border: 1px solid #003366;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <span style="display: inline-block; width: 8px; height: 8px;
                                     background: #00C853; border-radius: 50%;
                                     box-shadow: 0 0 4px #00C853; animation: capcom-pulse 2s ease-in-out infinite;"></span>
                        <span style="color: #003366; font-size: 14px; font-weight: 700;
                                     letter-spacing: 2px;">CAPCOM ONLINE</span>
                    </div>
                    <div style="color: #263238; font-size: 13px; font-family: 'SF Mono', 'Consolas', monospace;">
                        {_sid}<br/>
                        📸 {_sc} &nbsp; 📄 {_pc} &nbsp; 📊 {_dc}
                    </div>
                </div>
                <style>@keyframes capcom-pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.4; }}
                }}</style>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #f5f5f5; border-radius: 10px; padding: 10px 14px;
                            margin-bottom: 10px; border: 1px solid #ddd;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 8px; height: 8px;
                                     background: #bbb; border-radius: 50%;"></span>
                        <span style="color: #777; font-size: 14px; font-weight: 700;
                                     letter-spacing: 2px;">CAPCOM STANDBY</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        st.caption("ナビゲーション:\n1. Mission Control でデータをアップロードし、前処理を実行します。\n2. 上のリストから分析モジュールを選択します。")
        st.markdown("---")
        st.caption("© 2025-2026 しばやま")

# ==================================================================
# --- 5. Snapshot (VOYAGER連携) ---
# ==================================================================
def calculate_hhi(counts):
    """ヘルフィンダール・ハーシュマン指数 (HHI) を計算し、公取委基準で判定する (patiroha委譲)"""
    result = patiroha.calculate_hhi(counts)
    return result.value, result.status

def calculate_cagr_slope(df_subset, year_col='year'):
    """年平均成長率(CAGR)とトレンド(Slope)を計算する (patiroha委譲)"""
    result = patiroha.calculate_cagr(df_subset, year_col=year_col)
    return result.growth_rate, result.trend

@st.cache_data(show_spinner=False)
def generate_rich_summary(df_target, title_col='title', abstract_col='abstract', n_representatives=5):
    """
    VOYAGER用の高解像度サマリを生成する (Cached)
    - 統計情報 (HHI, CAGR, Trend)
    - 代表特許 (Centroid Distance)
    """
    summary = {
        "stats": {},
        "representatives": []
    }
    
    # 1. 統計情報の計算 (年次推移がある場合)
    if 'year' in df_target.columns:
        cagr, trend = calculate_cagr_slope(df_target)
        summary['stats']['cagr'] = f"{cagr:.1%}" if cagr is not None else "N/A"
        summary['stats']['trend'] = trend if trend else "N/A"

    # 2. HHI (市場集中度) の計算
    try:
        # 出願人情報 ('applicant_main') を利用して市場集中度を算出
        if 'applicant_main' in df_target.columns:
            all_apps = [a for sublist in df_target['applicant_main'] for a in sublist]
            counts = pd.Series(all_apps).value_counts().tolist()
            hhi, hhi_status = calculate_hhi(counts)
            summary['stats']['hhi'] = hhi
            summary['stats']['hhi_status'] = hhi_status
    except: pass
        
    # 3. 代表特許の抽出 (Centroid Distance)
    if 'sbert_embeddings' in st.session_state and not df_target.empty:
        try:
            # df_targetのindexを使ってembeddingsを抽出
            # 前提: df_mainのindexがresetされておらず、embeddingsと1対1対応していること
            valid_indices = [i for i in df_target.index if i < len(st.session_state.sbert_embeddings)]
            
            if valid_indices:
                vectors = st.session_state.sbert_embeddings[valid_indices]
                centroid = np.mean(vectors, axis=0)
                
                # 重心との距離計算 (Cosine Similarity相当)
                dots = np.dot(vectors, centroid)
                
                # 上位N件のインデックスを取得
                top_n_local_indices = np.argsort(dots)[::-1][:n_representatives]
                top_global_indices = [valid_indices[i] for i in top_n_local_indices]
                
                # データ取得
                reps = []
                invalid_count = 0
                
                # Column mapping for enhanced info
                col_map = st.session_state.get('col_map', {})
                app_col = col_map.get('applicant', 'applicant')
                ipc_col = col_map.get('ipc', None)
                num_col = col_map.get('pub_number', None) or col_map.get('app_num', None)

                for idx in top_global_indices:
                    try:
                        row = st.session_state.df_main.loc[idx]
                        t_val = str(row.get(title_col, ''))
                        a_val = str(row.get(abstract_col, ''))

                        # Enhanced Info
                        y_val = str(row.get('year', 'N/A'))
                        app_val = _extract_field_value(row, app_col, max_len=30) if app_col else "N/A"
                        ipc_val = _extract_field_value(row, ipc_col, max_len=40) if ipc_col else "N/A"
                        num_val = str(row.get(num_col, 'N/A')) if num_col and num_col in row.index else "N/A"

                        # Check validity
                        if (not t_val or t_val == 'nan') and (not a_val or a_val == 'nan'):
                             invalid_count += 1
                             title = "No Title"
                             abstract = "No Abstract"
                        else:
                             title = t_val if t_val and t_val != 'nan' else "No Title"
                             abstract = a_val if a_val and a_val != 'nan' else "No Abstract"

                        title = title.replace('\n', ' ')
                        abstract = abstract.replace('\n', ' ')[:200] + "..."

                        reps.append(f"- [{num_val}]【{title}】 (出願: {y_val}, {app_val}, IPC:{ipc_val}) {abstract}")

                        summary.setdefault('representatives_raw', []).append({
                            'title': title,
                            'abstract': abstract,
                            'year': y_val,
                            'applicant': app_val,
                            'ipc': ipc_val,
                            'number': num_val
                        })
                    except: pass
                
                # If mostly invalid, don't show
                if len(reps) > 0 and (invalid_count / len(reps)) > 0.5:
                     summary['representatives'] = [] # Suppress
                     summary['representatives_raw'] = []
                else:
                     summary['representatives'] = reps

        except Exception as e:
            summary['error'] = str(e)

    return summary

def get_cluster_representatives(df_subset, cluster_col='cluster', n_reps=3):
    """
    データフレーム内の各クラスタについて、重心に近い代表特許を抽出する (patiroha委譲)。
    Returns:
        dict: {cluster_id: ["- 【Title】(Applicant): Abstract...", ...]}
    """
    reps_dict = {}

    unique_clusters = sorted(df_subset[cluster_col].unique())
    embeddings = st.session_state.get('sbert_embeddings')

    col_map = st.session_state.get('col_map', {})
    title_col = col_map.get('title', 'title')
    abs_col = col_map.get('abstract', 'abstract')
    app_col = col_map.get('applicant', 'applicant')
    ipc_col = col_map.get('ipc', None)
    num_col = col_map.get('pub_number', None) or col_map.get('app_num', None)

    for cid in unique_clusters:
        if cid == -1: continue

        df_c = df_subset[df_subset[cluster_col] == cid]
        if df_c.empty: continue

        try:
            target_indices = []

            # patiroha.find_representatives でセントロイドベース抽出
            if embeddings is not None and len(embeddings) >= df_subset.index.max():
                valid_indices = [i for i in df_c.index if i < len(embeddings)]
                if valid_indices:
                    vectors = embeddings[valid_indices]
                    reps = patiroha.find_representatives(vectors, df_c, n=n_reps)
                    # reps から元のインデックスを復元
                    for r in reps:
                        if hasattr(r, 'index') and r.index is not None:
                            target_indices.append(r.index)

            # フォールバック: head選択
            if not target_indices:
                target_indices = df_c.index[:n_reps].tolist()

            cluster_reps = []
            for idx in target_indices:
                row = st.session_state.df_main.loc[idx]

                t_val = str(row.get(title_col, 'No Title')).replace('\n', ' ')
                a_val = str(row.get(abs_col, 'No Abstract')).replace('\n', ' ')[:200] + "..."
                app_val = _extract_field_value(row, app_col, max_len=30)
                y_val = str(row.get('year', 'N/A'))
                ipc_val = _extract_field_value(row, ipc_col, max_len=40) if ipc_col else "N/A"
                num_val = str(row.get(num_col, 'N/A')) if num_col and num_col in row.index else "N/A"

                cluster_reps.append(f"  * [{num_val}]【{t_val}】({app_val}, {y_val}, IPC:{ipc_val}): {a_val}")

            reps_dict[cid] = cluster_reps

        except Exception as e:
            continue

    return reps_dict


def _extract_field_value(row, col_name, max_len=30):
    """行からフィールド値を安全に抽出するヘルパー"""
    if not col_name or col_name not in row.index:
        return "N/A"
    val = row[col_name]
    if isinstance(val, list):
        clean_vals = [str(x).strip() for x in val if x and str(x).lower() != 'nan']
        result = ", ".join(clean_vals)
    elif pd.isna(val):
        return "N/A"
    else:
        result = str(val)
    if len(result) > max_len:
        result = result[:max_len] + "..."
    return result

def get_keyword_centric_representatives(df_target, top_keywords, n_reps=10):
    """
    ネットワークの主要キーワードを多く含む特許を抽出する (Keyword-Centric)。
    Args:
        df_target: 対象DataFrame
        top_keywords: ネットワークの中心性/頻度が高いキーワードリスト (list of str)
        n_reps: 抽出件数 (User request: significantly increase)
    Returns:
        list of dict: [{'title':..., 'applicant':..., 'abstract':..., 'score':...}]
    """
    if df_target.empty or not top_keywords:
        return []

    try:
        # Avoid side effects
        df = df_target.copy()
        
        col_map = st.session_state.get('col_map', {})
        t_col = col_map.get('title', 'title')
        a_col = col_map.get('abstract', 'abstract')
        app_col = col_map.get('applicant', 'applicant')
        ipc_col = col_map.get('ipc', None)
        num_col = col_map.get('pub_number', None) or col_map.get('app_num', None)
        y_col = 'year'

        # 上位30キーワードでスコアリング
        target_kws = top_keywords[:30]
        import re
        safe_kws = [re.escape(k) for k in target_kws if k and isinstance(k, str)]
        if not safe_kws: return []

        pattern = "|".join(safe_kws)

        # スコア計算（タイトル重み: x2, 要約: x1）
        score_series = pd.Series(0, index=df.index)
        if t_col in df.columns:
            score_series += df[t_col].astype(str).str.count(pattern) * 2
        if a_col in df.columns:
            score_series += df[a_col].astype(str).str.count(pattern)

        df['_kw_score'] = score_series

        # スコア > 0 のみ抽出してソート
        df_sorted = df[df['_kw_score'] > 0].sort_values(by='_kw_score', ascending=False).head(n_reps)

        reps = []
        for _, row in df_sorted.iterrows():
            title = str(row.get(t_col, 'No Title')).replace('\n', ' ')
            abstract = str(row.get(a_col, 'No Abstract')).replace('\n', ' ')[:200] + "..."
            year = str(row.get(y_col, '-'))

            app_val = "N/A"
            if app_col and app_col in row.index:
                val = row[app_col]
                if isinstance(val, list): app_val = ",".join([str(x) for x in val if x])
                else: app_val = str(val)
            if len(app_val) > 30: app_val = app_val[:30] + "..."

            # IPC
            ipc_val = "N/A"
            if ipc_col and ipc_col in row.index:
                val = row[ipc_col]
                if isinstance(val, list): ipc_val = ",".join([str(x) for x in val if x])[:40]
                else: ipc_val = str(val)[:40]

            # 番号
            num_val = str(row.get(num_col, 'N/A')) if num_col and num_col in row.index else "N/A"

            reps.append({
                'title': title,
                'abstract': abstract,
                'year': year,
                'applicant': app_val,
                'ipc': ipc_val,
                'number': num_val,
                'score': row['_kw_score'],
            })

        return reps

    except Exception as e:
        pass
        return []

def render_snapshot_button(title, description, key, fig=None, data_summary=None, figs=None):
    """
    グラフやデータをVOYAGER用に保存するボタンを表示する
    Args:
        title (str): スナップショットのタイトル
        description (str): スナップショットの説明
        key (str): 一意の識別キー
        fig (Figure): 単一のFigure (レガシーサポート)
        data_summary (dict): 分析データのサマリー
        figs (list): Figureのリスト (複数画像スナップショット用)
    """
    if 'snapshots' not in st.session_state:
        st.session_state['snapshots'] = []

    # 保存済みかチェック
    is_saved = any(s['id'] == key for s in st.session_state['snapshots'])
    
    btn_label = "📸 Snapshot Saved" if is_saved else "📸 Save Snapshot"
    btn_type = "primary" if not is_saved else "secondary"
    
    if st.button(btn_label, key=f"snap_btn_{key}", type=btn_type, disabled=is_saved):
        # モジュール名の決定 (data_summary['module']があれば優先)
        module_name = st.session_state.get('current_page', 'Unknown')
        if data_summary and isinstance(data_summary, dict) and 'module' in data_summary:
            module_name = data_summary['module']

        snapshot_data = {
            'id': key,
            'title': title,
            'description': description,
            'data_summary': data_summary,
            'module': module_name,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'images': [] # 画像リストを保存
        }
        
        # 対象フィギュアの統合
        target_figs = []
        if figs:
            target_figs = figs
        elif fig:
            target_figs = [fig]
            
        import io
        
        # --- ヘルパー: 1つのfigをバイト列に変換 ---
        def convert_fig_to_bytes(target_fig):
            if target_fig is None: return None, None
            img_bytes = None
            error_msg = None
            
            try:
                # Plotly
                if hasattr(target_fig, 'to_image'):
                    try:
                        # --- スマート解像度 & アスペクト比 ---
                        base_width = 1600
                        use_width = base_width
                        use_height = 1000 # デフォルトフォールバック
                        
                        is_saturn_v = module_name == 'Saturn V'
                        
                        try:
                            if is_saturn_v:
                                # SATURN V: アスペクト比を計算
                                xaxis = target_fig.layout.xaxis
                                yaxis = target_fig.layout.yaxis
                                if xaxis.range and yaxis.range:
                                    x_range = xaxis.range[1] - xaxis.range[0]
                                    y_range = yaxis.range[1] - yaxis.range[0]
                                    if x_range > 0 and y_range > 0:
                                        ratio = x_range / y_range
                                        calc_height = base_width / ratio
                                        calc_height = max(600, min(calc_height, 2400))
                                        use_height = int(calc_height)
                                    else:
                                        use_height = int(base_width * 0.618)
                                else:
                                    use_height = 1000
                            else:
                                # 標準ワイドフォーマット (16:9)
                                use_height = int(base_width * 9 / 16)
                        except:
                            use_height = 1000

                        img_bytes = target_fig.to_image(format="png", width=use_width, height=use_height, scale=3.0)
                    except Exception as e:
                        error_msg = f"Plotly Image Error (Kaleido): {str(e)}"
                
                # Matplotlib
                elif hasattr(target_fig, 'savefig'):
                    try:
                        buf = io.BytesIO()
                        target_fig.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        img_bytes = buf.getvalue()
                    except Exception as e:
                        error_msg = f"Matplotlib Image Error: {str(e)}"
                        
            except Exception as e:
                error_msg = f"General Image Error: {str(e)}"
                
            return img_bytes, error_msg

        # Process all figures
        for f in target_figs:
            ib, err = convert_fig_to_bytes(f)
            if ib:
                snapshot_data['images'].append(ib)
            if err:
                snapshot_data.setdefault('image_errors', []).append(err)
                st.warning(f"Image conversion warning: {err}")

        # レガシー互換性: メインの'image'を最初の有効な画像に設定
        if snapshot_data['images']:
            snapshot_data['image'] = snapshot_data['images'][0]
        else:
            snapshot_data['image'] = None

        st.session_state['snapshots'].append(snapshot_data)

        # --- CAPCOM出力フック ---
        try:
            import capcom
            if capcom.is_active():
                # PNG画像を保存
                for idx, img_bytes in enumerate(snapshot_data.get('images', [])):
                    if img_bytes:
                        suffix = idx if len(snapshot_data.get('images', [])) > 1 else None
                        capcom.save_snapshot_image(key, img_bytes, index=suffix)

                # 1スナップショット操作 = +1（画像数ではなく操作回数をカウント）
                capcom.increment_snap_count()

                # metadata.jsonを更新（全スナップショット）
                capcom.save_metadata(st.session_state['snapshots'])
        except Exception as e:
            pass

        st.rerun()

    if is_saved:
        st.success(f"'{title}' をVOYAGERポケットに保存しました")

# ==================================================================
# --- 5. AI アシスタント (共通) ---
# ==================================================================
def generate_ai_cluster_prompt(df_source, cluster_col, target_cols, tfidf_matrix, feature_names, n_samples=5):
    """クラスタごとの代表文献を抽出し、命名用プロンプトを生成する（c-TF-IDF方式）"""
    from sklearn.metrics.pairwise import euclidean_distances
    if df_source.empty: return "データがありません。"

    unique_clusters = sorted([c for c in df_source[cluster_col].unique() if c != -1])
    if not unique_clusters: return "有効なクラスタがありません。"

    # embeddingカラムの特定
    if 'umap_x' in df_source.columns and 'umap_y' in df_source.columns:
        embedding_cols = ['umap_x', 'umap_y']
    elif 'drill_x' in df_source.columns and 'drill_y' in df_source.columns:
        embedding_cols = ['drill_x', 'drill_y']
    elif 'acad_umap_x' in df_source.columns and 'acad_umap_y' in df_source.columns:
        embedding_cols = ['acad_umap_x', 'acad_umap_y']
    elif 'x' in df_source.columns and 'y' in df_source.columns:
        embedding_cols = ['x', 'y']
    else:
        return "埋め込み座標が見つかりません。"

    # c-TF-IDFでクラスタ判別キーワードを一括抽出
    ctfidf_keywords = {}
    try:
        # テキスト列を結合
        text_col_candidates = [c for c in target_cols if c and c in df_source.columns]
        if text_col_candidates:
            texts = df_source[text_col_candidates[0]].fillna('')
            for c in text_col_candidates[1:]:
                texts = texts + ' ' + df_source[c].fillna('')
            # patiroha.auto_label を top_n=10 で呼んでキーワード部分を抽出
            label_map_10 = patiroha.auto_label(
                texts, df_source[cluster_col].values,
                method='c-tfidf', top_n=10,
                label_format="{terms}",  # IDプレフィックスなし
            )
            ctfidf_keywords = label_map_10
    except Exception:
        pass

    sampled_docs = []

    for cid in unique_clusters:
        c_df = df_source[df_source[cluster_col] == cid]
        if c_df.empty: continue

        # c-TF-IDFキーワード
        keywords_str = ctfidf_keywords.get(cid, "")

        # 重心計算 → 重心に近い代表文献を抽出
        coords = c_df[embedding_cols].values
        centroid = coords.mean(axis=0)
        dists = euclidean_distances(coords, centroid.reshape(1, -1)).flatten()
        top_indices = np.argsort(dists)[:n_samples]

        docs = []
        for idx in top_indices:
            row = c_df.iloc[idx]
            text_parts = []
            for col in target_cols:
                if col and col in row and pd.notna(row[col]):
                    val = str(row[col]).replace('\n', ' ')
                    text_parts.append(val)
            docs.append(f"  - {' '.join(text_parts)}")

        sampled_docs.append(f"Cluster {cid}:\n[特徴語(c-TF-IDF)] {keywords_str}\n[代表文献]\n" + "\n".join(docs))

    sampled_docs_str = "\n\n".join(sampled_docs)

    prompt = f"""
あなたは熟練した特許情報アナリストです。
以下の「クラスタごとの特徴語と代表的特許リスト」を分析し、各クラスタの内容を端的に表す**「短い説明ラベル（日本語）」**を提案してください。

# 制約事項
- ラベルは**20文字以内**の日本語で記述してください。
- 専門用語を適切に使用し、技術的特徴や解決課題を反映させてください。
- 出力は **JSON形式のみ** としてください。解説は不要です。

# 出力例
{{
  "0": "全固体電池の固体電解質",
  "1": "画像認識による異常検知",
  "2": "カーボンニュートラル燃料製造"
}}

# 出力フォーマット (JSON)
{{
  "クラスタID (整数)": "提案ラベル",
  ...
}}

# クラスタデータ
{sampled_docs_str}
"""
    return prompt

def render_ai_label_assistant(df_source, cluster_col, label_map_key, col_map, tfidf_matrix, feature_names, widget_key_prefix=None):
    """AIラベルサジェストUI (共通部品)"""
    with st.expander("AIによるラベルサジェスト (オプション)"):
        st.markdown("LLM (ChatGPT等) にプロンプトを投げ、結果のJSONを取り込むことでラベルを自動設定します。")
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            n_samples_ai = st.number_input("1クラスタあたりのサンプル数", min_value=1, value=5, key=f"ai_n_samples_{label_map_key}")
        
        if st.button("プロンプトを生成", key=f"ai_gen_btn_{label_map_key}"):
            target_cols = [col_map.get('title'), col_map.get('abstract')]
            prompt = generate_ai_cluster_prompt(df_source, cluster_col, target_cols, tfidf_matrix, feature_names, n_samples=n_samples_ai)
            st.session_state[f"ai_prompt_{label_map_key}"] = prompt

        if f"ai_prompt_{label_map_key}" in st.session_state:
            st.code(st.session_state[f"ai_prompt_{label_map_key}"], language="markdown")
            st.info("👆 右上のコピーボタンでコピーし、LLMに入力してください。")

        st.markdown("---")
        st.markdown("**結果の取り込み (JSON)**")
        json_input = st.text_area("LLMの出力JSONを貼り付け:", height=150, key=f"ai_json_input_{label_map_key}")
        
        if st.button("サジェストを適用", key=f"ai_apply_btn_{label_map_key}"):
            try:
                # JSONのクリーニング (Markdownコードブロック除去)
                cleaned_json = re.sub(r'^```json\s*|\s*```$', '', json_input.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned_json)
                
                # key変換 (str -> int) & 適用
                current_map = st.session_state[label_map_key]
                count = 0
                for cid_str, label in data.items():
                    try:
                        cid = int(cid_str)
                        # df_sourceのクラスタカラムに存在するIDか確認
                        unique_cids = df_source[cluster_col].unique()
                        
                        if cid in current_map or cid in unique_cids: # 存在するクラスタのみ
                            new_val = f"[{cid}] {label}"
                            current_map[cid] = new_val
                            
                            # ウィジェットのステートも強制更新して、UI上の表示を同期させる
                            if widget_key_prefix:
                                w_key = f"{widget_key_prefix}_{cid}"
                                if w_key in st.session_state:
                                    st.session_state[w_key] = new_val
                            count += 1
                    except: pass
                
                # 反映 (session_stateのマップは参照渡しされている前提だが、念のため再代入)
                st.session_state[label_map_key] = current_map
                
                # [Saturn V] ラベルカラムの更新
                if label_map_key == "saturnv_labels_map" and 'df_main' in st.session_state:
                   # ラベル更新を反映
                   st.session_state.df_main['cluster_label'] = st.session_state.df_main['cluster'].map(current_map)
                elif label_map_key == "drill_labels_map" and 'df_drilldown_result' in st.session_state:
                   st.session_state.df_drilldown_result['drill_cluster_label'] = st.session_state.df_drilldown_result['drill_cluster'].map(current_map)

                # [MEGA] ラベルカラムの更新
                elif label_map_key == "mega_drill_labels_map" and 'df_drilldown' in st.session_state:
                   st.session_state.df_drilldown['label'] = st.session_state.df_drilldown['cluster_id'].map(current_map)
                   st.session_state.sbert_sub_cluster_map_auto = current_map

                st.success(f"{count} 件のラベルを更新しました！")
                st.rerun()
                
            except Exception as e:
                st.error(f"JSONパースエラー: {e}")

def create_label_editor_ui(original_map, current_map, key_prefix):
    """手動ラベル編集UI機能 (共通)"""
    widgets_dict = {}
    sorted_ids = sorted([cid for cid in original_map.keys() if cid != -1])
    for cluster_id in sorted_ids:
        orig_label = original_map.get(cluster_id, "")
        curr_label = current_map.get(cluster_id, orig_label)
        if orig_label == "(該当なし)": continue
        col1, col2 = st.columns([2, 3])
        with col1: st.markdown(f":green[{orig_label}]")
        with col2:
            key = f"{key_prefix}_{cluster_id}"
            if key not in st.session_state:
                st.session_state[key] = curr_label
            # value引数を指定せず、key経由でsession_stateの値を使用させる
            new_label = st.text_input(f"Edit {cluster_id}", label_visibility="collapsed", key=key)
            widgets_dict[cluster_id] = new_label
    if -1 in original_map:
        orig_noise = original_map[-1]
        curr_noise = current_map.get(-1, orig_noise)
        col1, col2 = st.columns([2, 3])
        with col1: st.markdown(f":green[{orig_noise}]")
        with col2:
            st.text_input(f"noise_label", value=curr_noise, disabled=True, key=f"{key_prefix}_noise")
            widgets_dict[-1] = curr_noise
    return widgets_dict

def render_cluster_dynamics_map(
    df, cluster_col, cluster_labels_map, year_col='year',
    cagr_window=5, unique_key='dynamics'
):
    """
    クラスタ動態マップを生成する。
    X=累積件数、Y=CAGR、バブル=シェア、4象限表示。

    Args:
        df: DataFrame with cluster and year columns
        cluster_col: cluster ID column name
        cluster_labels_map: dict mapping cluster_id -> label string
        year_col: year column name
        cagr_window: years for CAGR calculation (default 5)
        unique_key: unique key for Streamlit widgets

    Returns:
        tuple: (plotly Figure, dynamics_data dict for CAPCOM export)
    """
    import plotly.graph_objects as go
    import patiroha

    # Filter out noise (cluster == -1)
    df_valid = df[df[cluster_col] != -1].copy() if -1 in df[cluster_col].values else df.copy()

    clusters = sorted(df_valid[cluster_col].unique())
    if len(clusters) == 0:
        return None, None

    # Calculate per-cluster metrics
    total_patents = len(df_valid)
    years = df_valid[year_col].dropna()
    if years.empty:
        return None, None
    max_year = int(years.max())
    min_cagr_year = max_year - cagr_window

    cluster_data = []
    for cid in clusters:
        df_c = df_valid[df_valid[cluster_col] == cid]
        cumulative = len(df_c)
        share = cumulative / total_patents if total_patents > 0 else 0
        label = cluster_labels_map.get(cid, f"Cluster {cid}")

        # CAGR calculation
        years_c = df_c[year_col].dropna()
        if years_c.empty:
            cagr = 0.0
        else:
            recent = len(df_c[df_c[year_col] > min_cagr_year])
            past = len(df_c[df_c[year_col] <= min_cagr_year])
            if past > 0 and recent > 0:
                n_years = cagr_window
                cagr = (recent / past) ** (1 / n_years) - 1
            elif past == 0 and recent > 0:
                cagr = 1.0  # New cluster (100% growth)
            else:
                cagr = -0.5  # Declining

        cluster_data.append({
            'cluster_id': cid,
            'label': label,
            'cumulative_count': cumulative,
            'cagr': round(cagr, 4),
            'share': round(share, 4),
        })

    if not cluster_data:
        return None, None

    # Calculate thresholds (median)
    x_vals = [d['cumulative_count'] for d in cluster_data]
    y_vals = [d['cagr'] for d in cluster_data]
    import numpy as np
    x_threshold = float(np.median(x_vals))
    y_threshold = float(np.median(y_vals))

    # Assign quadrants
    quadrant_names = {
        (True, True): '成長リーダー',
        (False, True): '新興クラスタ',
        (True, False): '成熟クラスタ',
        (False, False): 'ニッチ/衰退',
    }
    for d in cluster_data:
        x_high = d['cumulative_count'] >= x_threshold
        y_high = d['cagr'] >= y_threshold
        d['quadrant'] = quadrant_names[(x_high, y_high)]

    # Build Plotly figure
    quadrant_colors = {
        '成長リーダー': '#2E7D32',
        '新興クラスタ': '#1565C0',
        '成熟クラスタ': '#E65100',
        'ニッチ/衰退': '#78909C',
    }

    fig = go.Figure()

    for qname, qcolor in quadrant_colors.items():
        q_items = [d for d in cluster_data if d['quadrant'] == qname]
        if not q_items:
            continue
        fig.add_trace(go.Scatter(
            x=[d['cumulative_count'] for d in q_items],
            y=[d['cagr'] for d in q_items],
            mode='markers+text',
            marker=dict(
                size=[max(15, d['share'] * 200) for d in q_items],
                color=qcolor,
                opacity=0.7,
                line=dict(width=1, color='white'),
            ),
            text=[d['label'] for d in q_items],
            textposition='top center',
            textfont=dict(size=10),
            name=qname,
            hovertemplate=(
                '<b>%{text}</b><br>'
                '累積件数: %{x}<br>'
                'CAGR: %{y:.1%}<br>'
                '<extra>%{fullData.name}</extra>'
            ),
        ))

    # Add quadrant lines
    fig.add_hline(y=y_threshold, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=x_threshold, line_dash="dash", line_color="gray", opacity=0.5)

    # Add quadrant labels
    x_range = max(x_vals) - min(x_vals) if len(x_vals) > 1 else max(x_vals)
    x_min_plot = min(x_vals) - x_range * 0.1
    x_max_plot = max(x_vals) + x_range * 0.1
    y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else abs(max(y_vals))
    y_min_plot = min(y_vals) - y_range * 0.15
    y_max_plot = max(y_vals) + y_range * 0.15

    annotations = [
        dict(x=x_max_plot * 0.95, y=y_max_plot * 0.95, text="成長リーダー", showarrow=False,
             font=dict(size=12, color='rgba(46,125,50,0.4)'), xanchor='right', yanchor='top'),
        dict(x=x_min_plot * 1.05 if x_min_plot > 0 else x_threshold * 0.1, y=y_max_plot * 0.95,
             text="新興クラスタ", showarrow=False,
             font=dict(size=12, color='rgba(21,101,192,0.4)'), xanchor='left', yanchor='top'),
        dict(x=x_max_plot * 0.95, y=y_min_plot * 1.05 if y_min_plot < 0 else y_threshold * 0.1,
             text="成熟クラスタ", showarrow=False,
             font=dict(size=12, color='rgba(230,81,0,0.4)'), xanchor='right', yanchor='bottom'),
        dict(x=x_min_plot * 1.05 if x_min_plot > 0 else x_threshold * 0.1,
             y=y_min_plot * 1.05 if y_min_plot < 0 else y_threshold * 0.1,
             text="ニッチ/衰退", showarrow=False,
             font=dict(size=12, color='rgba(120,144,156,0.4)'), xanchor='left', yanchor='bottom'),
    ]

    fig.update_layout(
        title=dict(text='クラスタ動態マップ', font=dict(size=16)),
        xaxis=dict(title='累積件数', gridcolor='#f0f0f0'),
        yaxis=dict(title=f'CAGR (直近{cagr_window}年)', tickformat='.0%', gridcolor='#f0f0f0'),
        plot_bgcolor='white',
        annotations=annotations,
        height=800,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    fig.update_layout(
        template=APOLLO_TEMPLATE,
        paper_bgcolor=APOLLO_BG,
    )

    # Build CAPCOM export data
    quadrant_summary = {}
    for qname in quadrant_colors:
        q_items = [d for d in cluster_data if d['quadrant'] == qname]
        quadrant_summary[qname] = {
            'count': len(q_items),
            'total_patents': sum(d['cumulative_count'] for d in q_items),
        }

    dynamics_data = {
        'x_axis': 'cumulative_count',
        'y_axis': 'cagr',
        'cagr_window_years': cagr_window,
        'x_threshold': x_threshold,
        'y_threshold': y_threshold,
        'clusters': cluster_data,
        'quadrant_summary': quadrant_summary,
    }

    # --- 象限別拡大図を生成 ---
    quadrant_figs = {}
    for qname, qcolor in quadrant_colors.items():
        q_items = [d for d in cluster_data if d['quadrant'] == qname]
        if not q_items:
            continue
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(
            x=[d['cumulative_count'] for d in q_items],
            y=[d['cagr'] for d in q_items],
            mode='markers+text',
            marker=dict(
                size=[max(20, d['share'] * 300) for d in q_items],
                color=qcolor,
                opacity=0.8,
                line=dict(width=1, color='white'),
            ),
            text=[d['label'] for d in q_items],
            textposition='top center',
            textfont=dict(size=11),
            name=qname,
            hovertemplate=(
                '<b>%{text}</b><br>'
                '累積件数: %{x}<br>'
                'CAGR: %{y:.1%}<br>'
                f'<extra>{qname}</extra>'
            ),
        ))
        fig_q.update_layout(
            title=dict(text=f'{qname} ({len(q_items)}クラスタ)', font=dict(size=14)),
            xaxis=dict(title='累積件数', gridcolor='#f0f0f0'),
            yaxis=dict(title=f'CAGR (直近{cagr_window}年)', tickformat='.0%', gridcolor='#f0f0f0'),
            plot_bgcolor='white',
            height=400,
            showlegend=False,
        )
        quadrant_figs[qname] = fig_q

    return fig, dynamics_data, quadrant_figs


def render_cluster_dynamics_section(
    df, cluster_col, cluster_labels_map, year_col='year',
    cagr_window=5, unique_key='dynamics',
    module_name='SaturnV'
):
    """
    クラスタ動態マップのUI全体を描画する。
    CAGRウィンドウ設定UI + 全体図 + 象限別拡大図 + スナップショット対応。
    """
    st.markdown("### クラスタ動態マップ")

    # --- CAGR ウィンドウ設定 ---
    years = df[year_col].dropna()
    if years.empty:
        st.info("年データがないためクラスタ動態マップを表示できません。")
        return None
    data_span = int(years.max()) - int(years.min())
    max_window = max(2, min(data_span, 15))
    default_window = min(cagr_window, max_window)

    cagr_window_actual = st.slider(
        "CAGR計算期間（直近N年）", min_value=2, max_value=max_window,
        value=default_window, key=f"cagr_window_{unique_key}",
        help=f"データ期間: {int(years.min())}〜{int(years.max())}年（{data_span}年間）"
    )

    result = render_cluster_dynamics_map(
        df, cluster_col, cluster_labels_map,
        year_col=year_col, cagr_window=cagr_window_actual,
        unique_key=unique_key,
    )
    if result is None or result[0] is None:
        st.info("クラスタ動態マップを生成できませんでした。")
        return None

    fig, dynamics_data, quadrant_figs = result

    # --- 全体図 ---
    st.plotly_chart(fig, use_container_width=True)

    # スナップショット（全体図）
    summary_lines = []
    for qname, qinfo in dynamics_data['quadrant_summary'].items():
        summary_lines.append(f"- {qname}: {qinfo['count']}クラスタ ({qinfo['total_patents']}件)")
    summary_text = "\n".join(summary_lines)

    render_snapshot_button(
        title=f"{module_name} クラスタ動態マップ",
        description=f"X=累積件数, Y=CAGR(直近{cagr_window_actual}年)\n{summary_text}",
        key=f"snap_{unique_key}_overview",
        fig=fig,
        data_summary=dynamics_data,
    )

    # --- 象限別拡大図 ---
    if quadrant_figs:
        with st.expander("象限別 拡大図", expanded=False):
            # 2列レイアウト
            cols = st.columns(2)
            quadrant_order = ['成長リーダー', '新興クラスタ', '成熟クラスタ', 'ニッチ/衰退']
            for i, qname in enumerate(quadrant_order):
                if qname in quadrant_figs:
                    with cols[i % 2]:
                        st.plotly_chart(quadrant_figs[qname], use_container_width=True)

                        # 象限内クラスタの一覧テーブル
                        q_items = [d for d in dynamics_data['clusters'] if d['quadrant'] == qname]
                        if q_items:
                            import pandas as pd
                            df_q = pd.DataFrame(q_items)[['label', 'cumulative_count', 'cagr', 'share']]
                            df_q.columns = ['クラスタ', '累積件数', 'CAGR', 'シェア']
                            df_q['CAGR'] = df_q['CAGR'].apply(lambda x: f"{x:.1%}")
                            df_q['シェア'] = df_q['シェア'].apply(lambda x: f"{x:.1%}")
                            st.dataframe(df_q, use_container_width=True, hide_index=True)

    return dynamics_data

def update_fig_layout(fig, title, height=1000, width=800, show_axes=False, show_legend=True):
    """Plotlyのレイアウトを統一的に更新する (APOLLO Standard 配色固定)"""
    # Sanitize title to remove implicit/explicit HTML tags (e.g. <b>)
    if isinstance(title, str):
        title = re.sub(r'<[^>]+>', '', title)

    layout_params = dict(
        template=APOLLO_TEMPLATE,
        title=dict(text=title, font=dict(size=18, color=APOLLO_TEXT, family="Helvetica Neue", weight="normal")),
        paper_bgcolor=APOLLO_BG,
        plot_bgcolor=APOLLO_BG,
        font=dict(color=APOLLO_TEXT, family="Helvetica Neue"),
        height=height,
        width=width,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#eee", borderwidth=1
        )
    )

    if not show_legend:
        layout_params['showlegend'] = False

    if not show_axes:
        layout_params['xaxis'] = dict(visible=False, showgrid=False, zeroline=False, showticklabels=False)
        layout_params['yaxis'] = dict(
            visible=False, showgrid=False, zeroline=False, showticklabels=False,
            scaleanchor="x", scaleratio=1
        )
    else:
        if "width" in layout_params:
            del layout_params["width"]

        layout_params['xaxis'] = dict(
            visible=True, showgrid=False, zeroline=False, showline=False, showticklabels=True
        )
        layout_params['yaxis'] = dict(
            visible=True, showgrid=True, gridcolor='#eee', zeroline=False, showline=False, showticklabels=True
        )

    fig.update_layout(**layout_params)
    return fig

# ==================================================================
# --- 5. 高度なテキスト処理 — patiroha委譲 ---
# ==================================================================

def normalize_text(text):
    """テキスト正規化 (patiroha委譲)"""
    return patiroha.normalize_text(text)

def apply_ngram_filters(text):
    """N-gramフィルタ適用 (patiroha委譲)"""
    return patiroha.apply_ngram_filters(text)

def load_tokenizer():
    """トークナイザ読み込み — patirohaが内部管理するため不要"""
    pass

def extract_keywords(text, tokenizer=None, stopwords=None, top_n=None, clean_html=False):
    """
    テキストから特徴語（名詞・複合名詞）を抽出する (patiroha委譲)
    Args:
        text (str): 対象テキスト
        tokenizer: 未使用 (後方互換性のため維持)
        stopwords (list/set): ストップワードのリスト
        top_n: 未使用 (後方互換性のため維持)
        clean_html (bool): Trueの場合、HTML/XMLタグを除去する (NPL推奨)
    Returns:
        list: 抽出されたキーワードのリスト
    """
    if stopwords is None:
        stopwords = patiroha.get_stopwords()
    return patiroha.extract_keywords(text, stopwords=stopwords, clean_html=clean_html)
