from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 반려묘를 양도할 때\n'
 '- 4. 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 보통약관 제24조(계약\n'
 '내용의 변경 등)에 따라 계약내용을 변경할 수 있습니다.<유의사항># [위험변경에 따른 계약변경 '
 '절차]![image](/image/placeholder)\n'
 '위험변경사항 통지(우편, 전화, 방문 등)\n'
 '↓\n'
 '계약자, 피보험자의 계약변경사항 확인 후 청약\n'
 '↓\n'
 '계약변경사항 인수 심사\n'
 '↓\n'
 '정산금액 처리(환급 또는추가납입)\n'
 '↓'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000489',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
