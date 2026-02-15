# sparse embedding model - simple version for inference only
from typing import List, Dict, Any
import math
from collections import defaultdict
import sys
import os
try:
    from tc_chunk import chunks
except Exception:
    chunks = []

# Initialize Kiwi tokenizer
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    KIWI_AVAILABLE = True
    print("✓ Kiwipiepy loaded successfully")
except ImportError:
    KIWI_AVAILABLE = False
    print("⚠ Kiwipiepy not available. Install: pip install kiwipiepy")

def tokenize_korean(text: str) -> List[str]:
    """
    Tokenize Korean text using Kiwi morphological analyzer
    Falls back to simple split if Kiwi is not available
    
    Args:
        text: Input text
    
    Returns:
        List of tokens (nouns and base forms)
    """
    if not KIWI_AVAILABLE:
        # Fallback to simple tokenization
        return text.lower().split()
    
    # Use Kiwi to extract morphemes
    result = kiwi.tokenize(text)
    tokens = []
    
    for token in result:
        # Each token has .form (surface form) and .tag (POS tag)
        pos = token.tag
        form = token.form.lower()
        
        # Include nouns and important POS tags
        if pos in ['NNG', 'NNP', 'NNB']:  # Common noun, Proper noun, Bound noun
            tokens.append(form)
        elif pos in ['VV', 'VA']:  # Verb, Adjective (base forms)
            tokens.append(form)
    
    return tokens

# # 사용자 입력: 청크 ID 또는 인덱스 선택
# chunk_input = input("\n보고 싶은 청크를 입력하세요 (ID: p3_c4 또는 인덱스: 0,1,2): ").strip()

# if chunk_input:
#     # 입력값 파싱 (ID 또는 인덱스, 쉼표로 여러 개)
#     inputs = [x.strip() for x in chunk_input.split(',')]
    
#     for inp in inputs:
#         found = False
        
#             # ID로 검색
#     for idx, chunk in enumerate(chunks):
#         if chunk.get('id') == inp:
#             found = True
#             print(f"\n{'='*100}")
#             print(f"[청크 {idx}] ID: {chunk.get('id')} | Page: {chunk.get('page')}")
#             print(f"{'='*100}")
#             print(f"{chunk.get('text', '')}")
#             break

#     if not found:
#         print(f"⚠️ 청크를 찾을 수 없음: '{inp}' (ID 또는 인덱스 확인)")

#     print("\n" + "=" * 100 + "\n")

# build TF-IDF model from corpus
def build_tfidf(chunks: List[Dict[str, Any]], predefined_words: List[str] = None):
    vocab: Dict[str, int] = {}
    df = defaultdict(int)
    docs_tokens: List[List[str]] = []

    # collect tokens and document frequencies from corpus
    for c in chunks:
        text_lower = c["text"].lower()
        # Use Korean tokenizer instead of simple split
        tokens = tokenize_korean(c["text"])
        
        # Add predefined words if found as substrings in chunk text
        if predefined_words:
            for pw in predefined_words:
                if pw.lower() in text_lower:
                    tokens.append(pw.lower())
        
        docs_tokens.append(tokens)
        for t in set(tokens):
            df[t] += 1
            if t not in vocab:
                vocab[t] = len(vocab)

    # ensure predefined words are present in vocab/df (df default 0)
    if predefined_words:
        for pw in predefined_words:
            pw_lower = pw.lower()
            if pw_lower not in vocab:
                vocab[pw_lower] = len(vocab)
            # ensure df key exists (leave count as-is if present)
            if pw_lower not in df:
                df[pw_lower] = 0

    N = len(chunks)
    idf = {t: math.log(N / (1 + df[t])) for t in df}

    sparse_vectors: Dict[str, List[tuple]] = {}
    for i, tokens in enumerate(docs_tokens):
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        vec = [(vocab[t], tf[t] * idf[t]) for t in tf if t in vocab]
        sparse_vectors[chunks[i]["id"]] = vec

    return sparse_vectors, vocab, idf

# convert query words to sparse vector
def query_to_sparse(words: List[str], vocab: Dict[str, int], idf: Dict[str, float]) -> List[tuple]:
    # words are predefined words that matched the query (substring-based)
    tf = defaultdict(int)
    for word in words:
        word_lower = word.lower()
        if word_lower in vocab:
            tf[word_lower] += 1

    query_vec = []
    for token, count in tf.items():
        idx = vocab[token]
        idf_score = idf.get(token, 0.0)
        tfidf_score = count * idf_score
        query_vec.append((idx, tfidf_score))

    return query_vec

# cosine similarity between two sparse vectors
def cosine_similarity(vec1: List[tuple], vec2: List[tuple]) -> float:
    d1 = {idx: val for idx, val in vec1}
    d2 = {idx: val for idx, val in vec2}
    dot = sum(d1.get(idx, 0) * val for idx, val in vec2)
    norm1 = math.sqrt(sum(v ** 2 for v in d1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in d2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# search top-3 chunks
def search_top3(query_vec: List[tuple], sparse_vectors: Dict[str, List[tuple]], chunks: List[Dict[str, Any]]) -> List[tuple]:
    scores = []
    for chunk in chunks:
        chunk_vec = sparse_vectors.get(chunk["id"], [])
        sim = cosine_similarity(query_vec, chunk_vec)
        scores.append((chunk["id"], sim, chunk.get("page"), chunk.get("text")[:100]))
    return sorted(scores, key=lambda x: -x[1])[:3]

# SPLADE (Sparse Lexical and Expansion Retrieval) based scoring - Non-normalized
def calculate_splade_weights(matched_query_words: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate non-normalized SPLADE weights for matched predefined words
    Weight = log(1 + IDF) ^ 2  (no normalization)
    
    Non-normalized approach:
    - Query score increases with more matched words
    - Each word contributes its full SPLADE weight
    - Useful for ranking by query term importance + quantity
    
    Args:
        matched_query_words: List of matched predefined words
        idf: IDF dictionary
    
    Returns:
        Dictionary of word -> non-normalized weight mapping
    """
    weights = {}
    
    # Calculate raw weights based on IDF (NO normalization)
    for word in matched_query_words:
        word_lower = word.lower()
        idf_value = idf.get(word_lower, 0.0)
        # Use log transformation of IDF for better scaling
        raw_weight = math.log(1.0 + idf_value) ** 2
        weights[word_lower] = raw_weight
    
    return weights

def calculate_predefined_word_score(matched_query_words: List[str], chunk_text: str, idf: Dict[str, float]) -> float:
    """
    Calculate SPLADE-based score for a chunk
    Score = Σ (TF × normalized_SPLADE_weight)
    
    Query gets SPLADE weights once, chunk scoring is based on word frequency matching
    """
    chunk_lower = chunk_text.lower()
    
    # Get SPLADE weights for matched words (normalized, sum=1)
    splade_weights = calculate_splade_weights(matched_query_words, idf)
    
    total_score = 0.0
    for word in matched_query_words:
        word_lower = word.lower()
        if word_lower in chunk_lower:
            # Calculate TF (term frequency in chunk)
            tf = chunk_lower.count(word_lower)
            
            # Get normalized SPLADE weight (from query)
            splade_weight = splade_weights.get(word_lower, 0.0)
            
            # Score: TF × SPLADE_weight
            word_score = tf * splade_weight
            total_score += word_score
    
    return total_score

# search top-3 chunks with SPLADE-based scoring
def search_top3_weighted(matched_query_words: List[str], chunks: List[Dict[str, Any]], idf: Dict[str, float], predefined_words_list: List[str]) -> List[tuple]:
    """
    Rank chunks using SPLADE algorithm
    - SPLADE weights computed once for query words
    - Chunk scores based on TF × normalized_SPLADE_weight
    - No cosine similarity or vector operations
    """
    scores = []
    for chunk in chunks:
        score = calculate_predefined_word_score(matched_query_words, chunk.get("text", ""), idf)
        scores.append((chunk["id"], score, chunk.get("page"), chunk.get("text")[:100]))
    
    # Filter out chunks with 0 score
    filtered_scores = [s for s in scores if s[1] > 0.0]
    
    if not filtered_scores:
        return []
    
    return sorted(filtered_scores, key=lambda x: -x[1])[:3]

# create TF-IDF bag of words vector (IDF score if word in query, 0 otherwise)
def tfidf_bag_of_words(query: str, words: List[str], vocab: Dict[str, int], idf: Dict[str, float]) -> List[float]:
    query_lower = query.lower()
    result = []
    
    for word in words:
        word_lower = word.lower()
        
        # Check if word is a substring of query (handles cases like "포도막염" in "포도막염에")
        if word_lower in query_lower:
            # Split predefined word into tokens
            word_tokens = word_lower.split()
            
            # Check if all tokens are in vocab
            all_tokens_in_vocab = all(token in vocab for token in word_tokens)
            
            if all_tokens_in_vocab:
                # Use average IDF score of all tokens
                idf_scores = [idf.get(token, 0.0) for token in word_tokens]
                avg_idf = sum(idf_scores) / len(idf_scores)
                result.append(avg_idf)
            else:
                result.append(0.0)
        else:
            result.append(0.0)
    return result

def match_predefined_words(query: str, words: List[str]) -> List[str]:
    """
    Match predefined words in query using substring matching
    Handles space variations (e.g., "심장 사상충" matches "심장사상충")
    """
    query_lower = query.lower()
    query_no_space = query_lower.replace(" ", "")  # Remove all spaces
    
    matched = []
    for word in words:
        word_lower = word.lower()
        word_no_space = word_lower.replace(" ", "")
        
        # Check both original and space-removed versions
        if word_lower in query_lower or word_no_space in query_no_space:
            matched.append(word)
    
    return matched

if __name__ == "__main__":
    # predefined bag of words
    predefined_words = ["슬관절탈구", "고관절탈구", "슬관절형성부전", "고관절형성부전", "대퇴 골두 허혈성 괴사", "감염병", "전염병", "세균감염", "바이러스감염", "기생충감염", "피부질환", "피부염", "알레르기", "알러지반응", "위장질환", "위염", "장염", "장질환", "간질환", "신장질환", "방광질환", "요로질환", "심장질환", "심부전", "판막질환", "호흡기질환", "폐질환", "기관지염", "폐렴", "신경계질환", "뇌질환", "경련", "발작", "종양", "양성종양", "악성종양", "암", "종괴", "종창", "골절", "탈구", "염좌", "인대손상", "근육손상", "디스크질환", "추간판탈출", "관절질환", "슬개골탈구", "고관절질환", "백내장", "녹내장", "안과질환", "치과질환", "구강질환", "치주질환", "치아손상", "췌장염", "당뇨병", "갑상선질환", "내분비질환", "면역질환", "자가면역질환", "빈혈", "혈액질환", "중독", "이물섭취", "교통사고상해", "낙상사고", "외상", "화상", "동상", "열사병", "탈수", "구토", "설사", "뒷다리 골육종", "기타 근골격 계통 양성 신생물", "기타 근골격 계통 악성 신생물", "기타 근골격 계통 신생물", "고관절 이형성", "고관절 탈구", "무혈성골두괴사", "슬개골 탈구", "십자 인대 손상 파열", "골절", "성장판 골절", "관절염", "퇴행성 관절염", "뼈연골", "근염", "염좌", "근골격계 질환", "눈 및 부속 기관 양성 신생물", "눈 및 부속 기관 악성 신생물", "눈 및 부속 기관 신생물", "안검 외반", "안검 내반", "안검염", "다래끼", "산립종", "마이봄선종", "체리아이", "제3안검 돌출", "비루관폐쇄", "유루증", "첩모난생", "첩모중생", "이소성첩모", "궤양성 각막염", "각막궤양", "각막 미란", "각막 이영양증", "각막염", "건성 각결막염", "결막염", "결막 부종", "포도막염", "홍채염", "전안방 출혈", "백내장", "수정체 탈구", "망막 변성", "망막 위축", "진행성 망막 위축", "망막 박리", "유리체 변성", "녹내장", "동양안충증", "초자체변성", "상공막염", "고양이 호산구성 각결막염", "눈곱", "결막 충혈", "눈 가려움", "순환기 계통 양성 신생물", "순환기 계통 악성 신생물", "순환기 계통 신생물", "고혈압", "저혈압", "부정맥", "판막 질환", "심부전", "심비대", "확장성", "심근병", "비대성", "제한성", "일시적 심근비대증", "심근증", "대동맥 협착증", "폐동맥 협착", "선천성 심장 질환", "심장사상충", "심혈관계 질환", "점액성 이첨판막변성", "신장 양성 신생물", "신장 악성 신생물", "신장 신생물", "이행상피세포암종", "방광 양성 신생물", "방광 악성 신생물", "방광 신생물", "비뇨기계 양성 신생물", "비뇨기계 악성 신생물", "비뇨기계 신생물", "신우 신염", "수신증", "신장 결석", "방광염", "방광 결석", "요도 폐색", "요로 결석증", "신경성 배뇨 이상", "비뇨기계 질환", "혈뇨", "요실금", "비정상 성분 소변", "핍뇨", "지방종", "조직구종", "유두종", "피지종", "모낭상피종", "기저세포종", "비만세포종", "악성 비만세포종", "흑색종", "피부 림프종", "편평세포암종", "항문주위선종", "항문주위선암종", "피부 신생물", "외이도염", "외이염", "중이염", "내이염", "농피증", "세균성 피부염", "말라세지아 피부염", "피부 사상균증", "곰팡이성 피부염", "모낭염", "모낭충증", "식이 알러지", "알러지 피부염", "아토피", "만성 피부염", "지루성 피부염", "피하 농양", "지방층염", "호산구성 육아종", "홍반루프스", "천포창", "지간 피부염", "족피부염", "꼬리샘 과증식", "발톱 주위염", "옴진드기", "개선충", "외부 기생충", "피부 질환", "귀 가려움", "발진", "피부염", "피부 가려움", "탈모", "선천적 질병", "유전적 질병", "파보 바이러스 감염", "디스템퍼 바이러스 감염", "파라인플루엔자 감염", "전염성 간염", "아데노 바이러스 2형 감염", "광견병", "코로나 바이러스 감염", "렙토스피라 감염", "필라리아 감염", "심장사상충 감염", "인플루엔자 감염", "상해", "질병", "예방접종", "치료", "검사", "투약", "예방접종", "정기검진", "예방검사", "임신", "출산", "제왕절개", "인공유산", "증상 치료", "중성화 수술", "불임 수술", "피임 수술", "미용 시술", "귀 성형", "꼬리 성형", "성대 제거", "미용성형 수술", "손톱 절제", "며느리발톱 제거", "잔존유치", "잠복고환", "제대허니아", "배꼽부위 탈장", "항문낭 제거", "외과수술", "점안", "귀청소", "첩모난생", "속눈썹 질환", "눈물샘 질환", "식이요법", "의약품 처방", "건강보조식품", "한방약", "한의학 치료", "침술", "인도의학", "허브요법", "아로마테라피", "대체의료", "재활치료", "목욕", "약욕", "처방샴푸", "기생충 제거", "벼룩 감염", "진드기 감염", "모낭충 감염", "기생충 질환", "안락사", "해부검사"]

    # build model for IDF calculation (vocab/idf needed for SPLADE scoring)
    if chunks:
        _, vocab, idf = build_tfidf(chunks, predefined_words)
    else:
        vocab = {word.lower(): i for i, word in enumerate(predefined_words)}
        idf = {word.lower(): 1.0 for word in predefined_words}
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Predefined words count: {len(predefined_words)}")
    
    # Option to view all vocab tokens
    view_vocab = input("\nView all vocab tokens? (y/n): ").strip().lower()
    if view_vocab == 'y':
        print("\n" + "="*100)
        print("[Vocab Tokens]")
        print("="*100)
        vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
        for word, idx in vocab_sorted:
            is_predefined = "✓" if word in [w.lower() for w in predefined_words] else " "
            print(f"[{is_predefined}] {idx:>4}: {word}")
        print("="*100)
    
    print()

    # user input: prefer command-line arg, then env QUERY, then interactive input
    if len(sys.argv) > 1:
        query_input = " ".join(sys.argv[1:]).strip()
    else:
        query_input = os.environ.get("QUERY")
        if query_input:
            query_input = query_input.strip()
        else:
            query_input = input("Enter query: ").strip()

    # Find matched predefined words in query
    matched_words = match_predefined_words(query_input, predefined_words)
    
    print(f"Query: '{query_input}'")
    print(f"Matched words: {matched_words}")
    print()
    
    # Calculate SPLADE scores for matched words
    if matched_words:
        splade_weights = calculate_splade_weights(matched_words, idf)
        
        print("[SPLADE Scores]")
        total_score = 0.0
        for word in matched_words:
            weight = splade_weights.get(word.lower(), 0.0)
            total_score += weight
            print(f"  - {word}: {weight:.4f}")
        
        print(f"\n[Query Total Score]: {total_score:.4f}")
    else:
        print("No matched predefined words found.")

    
