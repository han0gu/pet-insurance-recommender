from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가산이율 적용시 금융위원회 또는 금융감독원이 정당한 사유로 인정하는 경우에는<br>해당 기간에 대하여 가산이율을 적용하지 '
 '않습니다.<br>5'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000395',
              'chunk_char_len': 80,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
