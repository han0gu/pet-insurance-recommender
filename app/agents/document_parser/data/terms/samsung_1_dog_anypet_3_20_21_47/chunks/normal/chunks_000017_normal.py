from langchain_core.documents import Document

chunk = Document(
    page_content=('【핵연료물질】 사용된 연료를 포함합니다. 【핵연료물질에 의하여 오염된 물질】 원자핵 분열 생성물을 포함합니다.\n'
 '회사는 아래의 치료비 및 비용 또는 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000017',
              'chunk_char_len': 96,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
