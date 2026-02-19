from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 보험금의 지급\n'
 '제4조(보상하는 손해)\n'
 '① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 상해 또는 질병(이하 "사고"라 합니다)이 발생하여 그 치료를 직접적인 목적으로 '
 '국내에서 수의사에게 치료를 받은 때에는 피보험자가 부담 한 반려동물의 치료비를 이 약관에 따라 피보험자에게 치료비보험금으로 보상하여 '
 '드립니다. 단, 갱신계약의 경우에는 최초 보험가입시점 이후의 사고에 의한 경우에는 보험금을 지급합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
