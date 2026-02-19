from langchain_core.documents import Document

chunk = Document(
    page_content=('제3관 계약자의 계약 전 알릴 의무 등\n'
 '제12조(계약 전 알릴 의무)\n'
 '계약자, 피보험자 또는 이들의 대리인은 청약할 때 청약서(질문서를 포함합니다)에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 '
 '알려야 합니다.\n'
 '【관련법규】'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000035',
              'chunk_char_len': 131,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
