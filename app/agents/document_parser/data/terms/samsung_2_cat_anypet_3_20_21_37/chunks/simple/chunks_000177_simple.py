from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회사에 알리고 변 경된 장애기간이 기재된 장애인증명서를 '
 '제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 36},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000177',
              'chunk_char_len': 87,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
