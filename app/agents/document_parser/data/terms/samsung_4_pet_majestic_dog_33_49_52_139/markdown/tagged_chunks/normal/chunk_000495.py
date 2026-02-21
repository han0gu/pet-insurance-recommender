from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 이 특별약관에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와\n'
 '- 체결하고자 할 때 또는 이와 같은 계약이 있음을 알았을 때\n'
 '- 3. 반려견을 양도할 때\n'
 '- 4. 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 보통약관 제24조(계약\n'
 '내용의 변경 등)에 따라 계약내용을 변경할 수 있습니다.<유의사항># [위험변경에 따른 계약변경 '
 '절차]![image](/image/placeholder)\n'
 '위험변경사항 통지(우편, 전화, 방문 등)\n'
 '↓'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000495',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
