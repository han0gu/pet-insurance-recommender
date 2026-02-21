from langchain_core.documents import Document

chunk = Document(
    page_content=('- 있는 사람을 말합니다.\n'
 '- 3. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 드리는 증\n'
 '- 서를 말합니다.\n'
 '- 4. 진단계약: 계약을 체결하기 위하여 반려견이 건강진단을 받아야 하는 계약을 말합\n'
 '- 니다.\n'
 '- 5. 피보험자: 반려견의 소유와 관련하여 보험사고로 손해를 입은 사람을 말합니다.\n'
 '- 6. 반려견 : 보험증권에 기재된 반려견을 말하며, 이 특별약관에서 가입 가능한 반려\n'
 '- 견은 대한민국 내에서 피보험자와 거주를 함께하고 있는 개(犬)를말합니다. 다만'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000445',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
