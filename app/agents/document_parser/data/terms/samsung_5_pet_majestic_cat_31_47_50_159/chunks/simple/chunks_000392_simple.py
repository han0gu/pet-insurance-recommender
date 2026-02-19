from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조 (보험금을 지급하지 않는 사유)\n'
 '① 회사는 특별약관 일반사항 제7조(보험금을 지급하지 않는 사유)에서 정한 사유를 원인 으로 하여 생긴 손해는 보상하지 않습니다. ② '
 '회사는 다음 중 어느 한 가지 목적의 치료를 위한 상해 입원 수술비 또는 상해 통원 수술비에 대하여는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000392',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
