from langchain_core.documents import Document

chunk = Document(
    page_content='1. 계약자 또는 피보험자가 보험금을 지급받을 목적으로 고의로 보험금 지급사유를 발생시킨 경우',
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 52,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
