from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험기간 중에 보험계약자 또는 피보험자가<br>사고발생 위험이 현저하게 변경 또는 증가된 사실을 안 때에는 지체없이 '
 '보험자에게<br>통지하여야 하며, 위반 시 보험계약이 해지되거나 보험금 지급이 제한될 수 있습니<br>다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000264',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
