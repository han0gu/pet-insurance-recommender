from langchain_core.documents import Document

chunk = Document(
    page_content=('제 40조 (관할법원)\n'
 '이 계약에 관한 소송 및 민사조정은 계약자의 주소지를 관할하는 법원으로 합니다. 다만, 회사와 계약자가 합의하여 관할법원을 달리 정할 수 '
 '있습니다.\n'
 '제 41조 (소멸시효)\n'
 '보험금청구권, 만기환급금청구권, 보험료 반환청구권, 해약환급금 청구권, 계약자적립액 및 미경과보험료 반환청구권은 3년간 행사하지 않으면 '
 '소멸시효가 완성됩니다.\n'
 '<용어풀이>\n'
 '[소멸시효]\n'
 '- 47 -'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000163',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
