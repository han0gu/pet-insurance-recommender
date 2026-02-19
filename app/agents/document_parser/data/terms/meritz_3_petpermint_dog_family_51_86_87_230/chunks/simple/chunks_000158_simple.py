from langchain_core.documents import Document

chunk = Document(
    page_content=('【보험연도】\n'
 '당해 연도 보험계약 해당일부터 차년도 보험계약 해당일 전일까지 매1년 단위의 연도를 말합니다. 예를 들어, 보 험계약일이 2023년 4월 '
 '1일인 경우 보험연도는 4월 1일 부터 차년도 3월 31일까지 1년을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 82},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
