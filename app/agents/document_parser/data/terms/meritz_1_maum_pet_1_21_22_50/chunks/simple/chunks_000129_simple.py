from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료(계약의 청약을 권유하 기 위해 만든 자료 등을 말합니다)의 내용이 '
 '약관의 내용과 다른 경우에는 계약자에게 유리한 내용으로 계약이 성립된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 20},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000129',
              'chunk_char_len': 122,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
