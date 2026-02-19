from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조 (보험금을 지급하지 않는 사유)\n'
 '① 회사는 아래의 사유로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않습니다.\n'
 '1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, '
 '노동쟁의, 기타 이들과 유사한 사태 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변 4. 핵연료물질 또는 핵연료물질에 의하여 '
 '오염된 물질의 방사성, 폭발성 또는 그 밖의 유해한 특성 또는 이들 특성에 의한 사고 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 '
 '오염'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000711',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
