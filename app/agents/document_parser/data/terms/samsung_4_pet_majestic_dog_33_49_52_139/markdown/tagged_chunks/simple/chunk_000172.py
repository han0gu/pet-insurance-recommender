from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자의 의견에 따르기로 한 경우\n'
 '# <유의사항>\n'
 '분쟁조정은 이 약관의 (분쟁의 조정) 조항에 따라 금융감독원에 신청할 수 있습니다.③ 제2항에 의하여 장해지급률의 판정 및 지급할 '
 '보험금의 결정과 관련하여 확정된 장해\n'
 '지급률에 따른 보험금을 초과한 부분에 대한 분쟁으로 보험금 지급이 늦어지는 경우\n'
 '에는 보험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급합니다.<용어풀이># [장해지급률]질병이나 상해에 대하여 치유 후 남아있는 '
 '영구적인 장해에 의한 신체의 노동력 상실정도를 %로'),
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
 'indexing': {'chunk_id': 'chunk_000172',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
