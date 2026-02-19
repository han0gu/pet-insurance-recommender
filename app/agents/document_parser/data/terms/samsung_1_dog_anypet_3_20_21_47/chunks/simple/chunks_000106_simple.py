from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다. ③ 회사는 보상하지 않는 손해 등 계약자나 피보험자에게 '
 '불리하거나 부담을 주는 내용은 확대하여 해석하지 않습니다.\n'
 '제35조(설명서 교부 및 보험안내자료 등의 효력)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 135,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
