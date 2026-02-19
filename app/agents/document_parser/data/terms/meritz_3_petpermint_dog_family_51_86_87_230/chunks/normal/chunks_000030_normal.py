from langchain_core.documents import Document

chunk = Document(
    page_content=('【가지급보험금】\n'
 '보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회 사가 예상되는 보험금의 일부를 먼저 지급하는 제도로 피보험자 가 필요로 '
 '하는 비용을 보전해 주기 위해 회사가 먼저 지급하는 임시 교부금을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 57},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000030',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
