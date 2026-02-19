from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 회사가 보험금 지급사유 또는 보험료 납입면제 사유를 조사ㆍ확인하기 위해 필요한 기간이 제1항의 지급기일을 초과할 것이 명백히 '
 '예상되는 경우에는 그 구체적 사유와 지급예정일 및 보험금 가지급 제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대 하여 계약자, '
 '피보험자 또는 보험수익자에게 즉시 통지합니다. 다만, 지급예정일은 다 음 각 호의 어느 하나에 해당하는 경우를 제외하고는 제9조(보험금 '
 '등의 청구)에서 정 한 서류를 접수한 날부터 30영업일 이내에서 정합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
