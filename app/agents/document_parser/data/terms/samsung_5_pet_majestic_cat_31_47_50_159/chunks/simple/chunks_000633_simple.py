from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약이 다음 각 호의 조건을 충족하고 계약자가 제4항에 따라 재가입의사를 표시한 때에는 특별약관 일반사항의 제19조(특별약관의 성립) '
 '및 제21조(약관 교부 및 설명 의무 등)를 준용하여 회사가 정한 절차에 따라 계약자는 기존 계약에 이어 재가입할 수 있으며, 이 경우 '
 '회사는 기존계약의 가입 이후 발생한 반려동물의 상해 또는 질병 을 사유로 가입을 거절할 수 없습니다. 단, 특별약관 일반사항의 '
 '제19조(특별약관의 성립) 제1항 및 제2항에도 불구하고 제2항에서 말하는 별도의 반려동물보험 상품으로 체결될 수 있습니다. 1'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000633',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
