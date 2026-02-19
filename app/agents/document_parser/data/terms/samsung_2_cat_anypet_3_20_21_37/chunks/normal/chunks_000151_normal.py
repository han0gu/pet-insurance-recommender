from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(자료의 제출 및 열람)\n'
 '① 계약자는 계약이 효력상실 또는 해지된 경우에는 효력상실 또는 해지일까지의 보험료를 확정하기 위하여 필요한 서류를 효력상실 또는 해지 '
 '즉시 회사에 제출해야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 30},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 109,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
