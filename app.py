import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps
from google import genai

st.set_page_config(page_title="教科書レジュメ生成アプリ", layout="centered")

st.title("📖 教科書・ノートのスマートレジュメ作成アプリ")
st.write("写真をアップロードすると、文字を認識し、Geminiが図表付きのわかりやすい学習用レジュメにまとめ直します。")

# APIキーの取得
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None
    st.warning("⚠️ 管理者設定: Streamlit Cloudの Secrets に GEMINI_API_KEY を設定してください。")

# 最大10枚までアップロードできるように設定
uploaded_files = st.file_uploader(
    "教科書やノートの画像を選択、または撮影してください (最大10枚まで)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    # アップロード枚数が10枚を超えた場合の制限
    if len(uploaded_files) > 10:
        st.error("⚠️ 一度にアップロードできる画像は最大10枚までです。選択し直してください。")
    else:
        st.success(f"📸 {len(uploaded_files)} 枚の画像が読み込まれました。")
        
        # 選択されたすべての画像を並べてプレビュー表示
        cols = st.columns(min(len(uploaded_files), 3)) # 最大3列でプレビューを表示
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                img_preview = Image.open(file)
                img_preview = ImageOps.exif_transpose(img_preview).convert("RGB")
                st.image(img_preview, caption=f"ページ {idx + 1}", use_container_width=True)

        if st.button("✨ レジュメを生成する"):
            if not api_key:
                st.error("APIキーが設定されていません。アプリの設定からSecretsを登録してください。")
            else:
                # 進行状況バーを設定
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_ocr_texts = []
                
                try:
                    # 1. すべての画像から1枚ずつOCRを実行
                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"📖 画像 {idx + 1}/{len(uploaded_files)} 枚目を処理中...")
                        
                        image = Image.open(file)
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        
                        img = np.array(image)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        
                        ocr_text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 6")
                        if ocr_text.strip():
                            all_ocr_texts.append(f"--- [ページ {idx + 1} の内容] ---\n{ocr_text}")
                        
                        # 進捗率の更新
                        progress_bar.progress(int((idx + 1) / len(uploaded_files) * 50))
                    
                    if not all_ocr_texts:
                        st.warning("画像から文字を検出できませんでした。画像を直接Geminiに渡すなどして試行します。")
                        combined_text = "(画像からの文字検出なし)"
                    else:
                        combined_text = "\n\n".join(all_ocr_texts)
                    
                    # 2. Geminiによる要約レジュメ一括生成
                    status_text.text("🤖 Geminiが全体のレジュメを作成中... (約10〜15秒)")
                    progress_bar.progress(75)
                    
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    提供された複数のOCRテキスト（教科書やノートの複数ページ分の読み取りデータ）を元に、全体の流れが一目で理解できるように構造化した、高品質な「1つの統合レジュメ（要約学習プリント）」を作成してください。

                    【レジュメ作成ルール】
                    1. **全体像・タイトル**: 
                       全体を包摂するわかりやすい大タイトルと導入を書いてください。
                    2. **各ページ/テーマごとの要点整理**:
                       複数ページにまたがっているため、テーマの変遷や繋がりがわかるようにセクション分けしてください。
                    3. **視覚的な整理（表・グラフ）の積極的導入**:
                       - 比較、対比、分類、タイムライン、対照事項がある場合は、必ずMarkdownの「表（テーブル）」形式に整理してください。
                       - プロセス、順序（ステップ）、因果関係、システムの関係性がある場合は、Markdown内でレンダリング可能な「Mermaid.js (graph TDなど)」コードブロックを使用してください。
                    4. **重要キーワード集**: 専門用語をピックアップし、初心者向けに優しく解説してください。
                    5. **まとめ**: 最後に最も覚えるべき要点を簡潔な箇条書きでまとめてください。

                    【対象テキスト群】
                    {combined_text}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt,
                    )
                    
                    # 進捗完了
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    # 結果表示
                    st.success("🎉 全ページの統合レジュメ作成が完了しました！")
                    st.subheader("📚 作成された学習レジュメ")
                    st.markdown(response.text)
                    
                except Exception as e:
                    status_text.empty()
                    st.error(f"エラーが発生しました: {e}")
