from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지나는 동안 보장이 제외되는 질병으로 추가 진단(단순 건강검진 제외) 또는 치료 사\n'
 '- 실이 없을 경우, 청약일로부터 5년이 지난 이후에는 이 약관에 따라 보장합니다.\n'
 '- ⑥ 제5항의 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없는 경우는 다음 각 호의\n'
 '- 경우를 포함합니다.\n'
 '- 1. 검진결과 추가검사 또는 치료가 필요하지 않았던 경우\n'
 '- 2. 부담보가 지정된 질병 또는 증상이 악화되지 않고 유지된 경우\n'
 "- ⑦ 제5항의 '청약일로부터 5년이 지나는 동안' 이라 함은 제27조(보험료의 납입이 연체"),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000057',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
