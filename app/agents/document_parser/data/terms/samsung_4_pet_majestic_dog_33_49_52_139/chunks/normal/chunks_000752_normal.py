from langchain_core.documents import Document

chunk = Document(
    page_content=('보상하지 않습니다.\n'
 '1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, '
 '노동쟁의, 기타 이들과 유사한 사태 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변 4. 핵연료물질 또는 핵연료물질에 의하여 '
 '오염된 물질의 방사성, 폭발성 또는 그 밖의 유해한 특성 또는 이들 특성에 의한 사고 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 '
 '오염\n'
 '<용어풀이>\n'
 '[핵연료물질]\n'
 '사용된 연료를 | 포함합니다.\n'
 '[핵연료물질에 | 의하여 오염된 물질]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 121},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000752',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
