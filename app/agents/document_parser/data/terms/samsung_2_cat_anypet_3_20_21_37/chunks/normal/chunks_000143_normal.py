from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자는 매월 10일까지 전월말까지의 피보험자수에 관한 서류를 회사에 제출하여야 합니다. 그 러나 계약이 효력상실 또는 해지된 '
 '경우에는 효력상실 또는 해지일까지의 보험료를 확정하기 위하여 필요한 서류를 효력상실 또는 해지 즉시 회사에 제출하여야 합니다. 2. '
 '회사는 보험기간중이나 보험기간 만료후 보험료를 산출하기 위하여 필요하다고 인정될 경우에 는 계약자의 서류를 열람할 수 있습니다. 3. '
 '회사는 보험기간 만료와 동시에 제1호에의한 피보험자수에 따라 산출된 확정보험료와 기납입한 보험료를 비교하여 그 차액을 정산합니다. 4'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 29},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
