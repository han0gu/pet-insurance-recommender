from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회사에 알리고 변 경된 장애기간이 기재된 장애인증명서를 '
 '제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000220',
              'chunk_char_len': 87,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
