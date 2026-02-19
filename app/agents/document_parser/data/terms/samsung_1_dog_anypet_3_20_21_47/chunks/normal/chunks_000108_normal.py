from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료의 내용이 약관의 내용과 다른 경 우에는 계약자에게 유리한 내용으로 '
 '계약이 성립된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 91,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
