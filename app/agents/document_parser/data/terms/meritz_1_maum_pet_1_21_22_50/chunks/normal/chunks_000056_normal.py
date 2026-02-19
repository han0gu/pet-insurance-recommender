from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때 2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 '
 '계약을 다른 보험자와 체결하 고자 할 때 또는 이와 같은 계약이 있음을 알았을 때 3. 반려동물을 양도할 때 4. 위 이외에 위험이 '
 '뚜렷이 변경되거나 변경되었음을 알았을 때'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
