# textbook-ocr-gemini

import streamlit as st
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps
from google import genai
import os

# クラウド上(Linux)のTesseractパスが自動解決しない場合のフォールバック
# (Streamlit Cloudでは、下記の「packages.txt」を設定すると自動でインストールされます)

st.set_page_config(page_title="教科書レジュメ生成アプリ", layout="centered")

st.title("📖 教科書・ノートのスマートレジュメ作成アプリ")
st.write("写真をアップロードすると、文字を認識し、Geminiが図表付きのわかりやすい学習用レジュメにまとめ直します。")

# APIキーの取得 (Streamlitのシークレットから安全に取得)
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None
    st.warning("⚠️ 管理者設定: Streamlit Cloudの Secrets に GEMINI_API_KEY を設定してください。")

uploaded_file = st.file_uploader("教科書やノートの画像を選択、または撮影してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    if st.button("✨ レジュメを生成する"):
        if not api_key:
            st.error("APIキーが設定されていません。アプリの設定からSecretsを登録してください。")
        else:
            with st.spinner("🤖 画像の解析とレジュメ生成を行っています... (約10〜15秒)"):
                try:
                    # 1. OCR処理
                    img = np.array(image)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    
                    ocr_text = pytesseract.image_to_string(processed, lang="jpn", config="--psm 6")
                    
                    if not ocr_text.strip():
                        st.warning("画像から文字を検出できませんでした。そのまま画像をGeminiに解釈させます。")
                    
                    # 2. Geminiによる教材生成
                    client = genai.Client(api_key=api_key)
                    prompt = f"""
                    提供されたテキスト（教科書の画像から読み取ったデータ）を元に、学生が直感的に理解できる高品質な「レジュメ（要約学習プリント）」を作成してください。

                    【レジュメ作成ルール】
                    1. **タイトルと概要**: 何について書かれた内容なのか、一目でわかるタイトルと短い導入を書いてください。
                    2. **視覚的な整理（表・グラフ）の積極的導入**:
                       - 比較、対比、分類、タイムラインなどがある場合は、必ずMarkdownの「表（テーブル）」形式に整理してください。
                       - 流れ（フロー）、構成要素のつながりなどは、Markdown内の「Mermaid.js (graph TDなど)」コードブロックを使って図解を作成してください。
                    3. **重要キーワード解説**: 専門用語をいくつかピックアップし、初心者向けに優しく解説してください。
                    4. **まとめ**: 覚えるべきポイントを最後に箇条書きでまとめてください。

                    【対象テキスト】
                    {ocr_text}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-3.8-flash',
                        contents=prompt,
                    )
                    
                    # 結果表示
                    st.success("🎉 レジュメの作成が完了しました！")
                    st.subheader("📚 作成された学習レジュメ")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
