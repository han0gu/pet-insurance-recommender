from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 따라 위험이 감소된 경우에는 그 차액보험료를 돌려드리며, 위험이 증가된 경우에 는 통지를 받은 날부터 1개월 이내에 '
 '보험료의 증액을 청구하거나 계약을 해지할 수 있습니다. ③ 계약자 또는 피보험자는 주소 또는 연락처가 변경된 경우에는 지체없이 이를 '
 '회사에 알려야 합니 다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 10},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 162,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
