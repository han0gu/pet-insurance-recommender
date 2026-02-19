from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 회사는 제1항에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약관을 교부하고 변경 된 계약자가 요청하는 경우 약관의 '
 '중요한 내용을 설명하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
