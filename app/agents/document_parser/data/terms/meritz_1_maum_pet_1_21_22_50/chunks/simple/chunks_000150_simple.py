from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 제6조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 교부하고, 그 서류를 접수받은 후 지체없이 지급할 보험금을 '
 '결정하고 지급할 보험금이 결정되면 7 일 이내에 이를 지급하여 드립니다. 또한, 지급할 보험금이 결정되기 전이라도 피보험 자의 청구가 '
 '있을 때에는 회사가 추정한 보험금의 50% 상당액을 가지급보험금으로 지 급합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000150',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
