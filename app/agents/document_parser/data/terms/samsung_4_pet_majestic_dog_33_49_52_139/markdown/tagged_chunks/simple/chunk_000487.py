from langchain_core.documents import Document

chunk = Document(
    page_content=('급 제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자 또는 보험수\n'
 '익자에게 즉시 통지합니다. 다만, 지급예정일은 다음 각 호의 어느 하나에 해당하는\n'
 '경우를 제외하고는 제8조(보험금의 청구)에서 정한 서류를 접수한 날부터 30영업일\n'
 '이내에서 정합니다.- \n'
 '- 1. 소송제기\n'
 '- 2. 분쟁조정 신청\n'
 '- 3. 수사기관의 조사\n'
 '- 4. 해외에서 발생한 보험사고에 대한 조사\n'
 '- 5. 제6항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보험수'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000487',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
