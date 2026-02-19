from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[핵연료물질]\n'
 '사용된 연료를 | 포함합니다.\n'
 '[핵연료물질에 | 의하여 오염된 물질]\n'
 '원자핵 분열 생성물을 포함합니다.\n'
 '6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해 7. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 상해 및 질병 8. '
 '반려묘를 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로 이 용함으로써 발생한 손해 9. 원인이 어떠한 경우에도 '
 '반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에\n'
 '대한 태만'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 108},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000659',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
