from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 지급예정일은 다음 각 호의 어느 하나에 해당하는 경우를 제외하고는 제8조(보험금의 청구)에서 정한 서류를 접수한 날부터 '
 '30영업일 이내에서 정합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 7},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
