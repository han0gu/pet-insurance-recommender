from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 검진결과 추가검사 또는 치료가 필요하지 않았던 경우 2. 부담보가 지정된 질병 또는 증상이 악화되지 않고 유지된 경우\n'
 '⑦ 제5항의 ‘청약일로부터 5년이 지나는 동안’이라 함은 제30조(보험료의 납입이 연체 되는 경우 납입최고(독촉)와 계약의 해지)에서 '
 '정한 계약의 해지가 발생하지 않은 경 우를 말합니다. ⑧ 제31조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에서 정한 '
 '계약의 부 활이 이루어진 경우 부활을 청약한 날을 제5항의 청약일로 하여 적용합니다.\n'
 '제21조 (청약의 철회)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
