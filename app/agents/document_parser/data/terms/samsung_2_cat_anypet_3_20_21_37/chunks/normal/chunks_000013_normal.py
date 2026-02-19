from langchain_core.documents import Document

chunk = Document(
    page_content=('② 반려동물이 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터 180일 이내의 치료비는 보상하여 드립니다. 다만, '
 '사고일 또는 발병일부터 365일이내의 치료인 경우에 한합니다.\n'
 '제5조(보상하지 않는 손해)\n'
 '① 회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
