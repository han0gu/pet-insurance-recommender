from langchain_core.documents import Document

chunk = Document(
    page_content=('제10조(사기에 의한 계약)\n'
 '\uf000 계약자 또는 피보험자가 사기에 의하여 계약이 성립되었 음을 회사가 증명하는 경우에는 계약일부터 5년 이내(사기 사실을 안 '
 '날부터 1개월 이내)에 계약을 취소할 수 있습니 다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000226',
              'chunk_char_len': 115,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
