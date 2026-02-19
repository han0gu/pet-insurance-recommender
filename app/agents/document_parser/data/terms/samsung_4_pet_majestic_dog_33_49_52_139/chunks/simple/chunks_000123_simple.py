from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 제16조(계약 전 알릴 의무)에 따라 계약자 또는 피보험자가 회사에 알린 내용이나 건강진단 내용이 보험금 지급사유의 발생에 영향을 '
 '미쳤음을 회사가 증명하는 경 우 2. 제18조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는 경우 3. '
 '진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우. 다만, 진 단계약에서 진단을 받지 않은 경우라도 상해로 보험금 '
 '지급사유가 발생하는 경우 에는 보장을 해드립니다.\n'
 '제28조 (제2회 이후 보험료의 납입)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 44},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000123',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
