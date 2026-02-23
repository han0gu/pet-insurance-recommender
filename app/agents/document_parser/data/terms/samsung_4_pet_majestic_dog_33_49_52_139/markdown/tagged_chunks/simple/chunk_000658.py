from langchain_core.documents import Document

chunk = Document(
    page_content=('한도 내에서 아래의 권리를 가집니다. 다만, 회사가 보상한 금액이 피보험자가 입은\n'
 '손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가\n'
 '집니다.- 1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권\n'
 '- 2. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 있을 경우에는 그 대위권\n'
 '<예시안내>- 122 -제3자의 귀책사유로 손해가 발생한 상황에서 회사가 1,000만원의 보험금을 지급했다면, 회사는'),
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
 'indexing': {'chunk_id': 'chunk_000658',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
