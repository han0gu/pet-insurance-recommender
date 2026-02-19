from langchain_core.documents import Document

chunk = Document(
    page_content=('이자를 더하여 지급하지 않습니다.\n'
 '⑥ 회사가 제3항에 따라 일부보장 제외 조건을 붙여 승낙하였더라도 청약일로부터 5년\n'
 '(갱신계약의 경우에는 최초계약 청약일로부터 5년)이 지나는 동안 보장이 제외되는 질 병으로 추가 진단(단순 건강검진 제외) 또는 치료 '
 '사실이 없을 경우, 청약일로부터 5 년이 지난 이후에는 이 약관에 따라 보장합니다.\n'
 '⑦ 제6항의 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없는 경우는 다음 각 호의 경우를 포함합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 51},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000217',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
