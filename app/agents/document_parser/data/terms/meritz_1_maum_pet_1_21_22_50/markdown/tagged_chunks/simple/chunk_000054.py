from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와 체결하\n'
 '- 고자 할 때 또는 이와 같은 계약이 있음을 알았을 때\n'
 '- 3. 반려동물을 양도할 때\n'
 '- 4. 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '- 9 -② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 제23조(계약내용의 변\n'
 '경 등)에 따라 계약내용을 변경할 수 있습니다.![image](/image/placeholder)\n'
 '[위험변경에 따른 계약변경 절차]\n'
 '위험변경사항 통지\n'
 '(우편, 전화, 방문 등)\n'
 '↓'),
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
 'indexing': {'chunk_id': 'chunk_000054',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
