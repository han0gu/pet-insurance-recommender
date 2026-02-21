from langchain_core.documents import Document

chunk = Document(
    page_content=('발생하여 그 치료를 직접적인 목적으로 국내에서 수의사에게 치료를 받은 때에는 피보험자가 부담\n'
 '한 반려동물의 치료비를 이 약관에 따라 피보험자에게 치료비보험금으로 보상하여 드립니다. 단,\n'
 '갱신계약의 경우에는 최초 보험가입시점 이후의 사고에 의한 경우에는 보험금을 지급합니다.- 5 -당신에게 좋은보험 삼성화재② 반려동물이 '
 '제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터 180일 이내의'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 227,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
