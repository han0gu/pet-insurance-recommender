from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그 밖의\n'
 '- 유해한 특성 또는 이들 특성에 의한 사고\n'
 '- 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '<용어풀이>[핵연료물질]사용된 연료를 포함합니다.\n'
 '[핵연료물질에 의하여 오염된 물질]\n'
 '원자핵 분열 생성물을 포함합니다.6. 반려묘를 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로 이# 용함으로써 '
 '발생한 손해7. 수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인한'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000578',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
