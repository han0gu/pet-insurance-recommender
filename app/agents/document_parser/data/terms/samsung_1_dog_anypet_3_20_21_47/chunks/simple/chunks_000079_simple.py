from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약자 또는 피보험자가 최초 계약 청약시(2회 이상 부활이 이루어진 경우 종전 '
 '모든 부활 청약 포함) 제12조(계약 전 알릴의무)를 위반한 경우에 는 제26조(계약의 해지) 제3항이 적용됩니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000079',
              'chunk_char_len': 139,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
