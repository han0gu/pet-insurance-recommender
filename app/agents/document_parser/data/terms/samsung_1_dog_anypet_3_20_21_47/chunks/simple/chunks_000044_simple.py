from langchain_core.documents import Document

chunk = Document(
    page_content=('제14조(사기에 의한 계약)\n'
 '계약자, 피보험자 또는 이들의 대리인의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에 는 계약일부터 5년 이내(사기사실을 안 '
 '날부터 1개월 이내)에 계약을 취소할 수 있습니다.\n'
 '제4관 보험계약의 성립과 유지\n'
 '제15조(보험계약의 성립)'),
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
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
