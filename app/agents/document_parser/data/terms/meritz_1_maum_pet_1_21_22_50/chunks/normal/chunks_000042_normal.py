from langchain_core.documents import Document

chunk = Document(
    page_content=('【가지급보험금】\n'
 '보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회사가 예상되는 보험 금의 일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 '
 '비용을 보전해 주기 위 해 회사가 먼저 지급하는 임시 교부금을 말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
