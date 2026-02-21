from langchain_core.documents import Document

chunk = Document(
    page_content=('- 년이 지난 이후에는 이 약관에 따라 보장합니다.\n'
 '- ⑥ 제5항의 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없는 경우는 다음 각 호의\n'
 '- 경우를 포함합니다.\n'
 '- 1. 검진결과 추가검사 또는 치료가 필요하지 않았던 경우\n'
 '- 2. 부담보가 지정된 질병 또는 증상이 악화되지 않고 유지된 경우\n'
 '- ⑦ 제5항의 ‘청약일로부터 5년이 지나는 동안’이라 함은 제30조(보험료의 납입이 연체\n'
 '- 되는 경우 납입최고(독촉)와 계약의 해지)에서 정한 계약의 해지가 발생하지 않은 경\n'
 '- 우를 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000076',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
