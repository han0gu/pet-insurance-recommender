from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항의 경우에 회사는 청약서를 접수한 날로부터 30일 이내에 승낙 또는 거절하여야 하며, 승낙 한 때에는 지정계좌에서 제1회 '
 '보험료를 받고 보험증권을 교부합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
