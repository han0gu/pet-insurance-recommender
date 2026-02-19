from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 위생관리, 미모를 위한 성형수술. 다만, 사고전 상태로의 회복을 위한 수술은 보상 합니다. 5. 선천적 기형 및 이에 근거한 병상\n'
 '제7조 (특별약관의 소멸)\n'
 '피보험자가 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시 이 '
 '특별약관의 계약자적립액 및 미경과보험료 를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 76},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000399',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
