from langchain_core.documents import Document

chunk = Document(
    page_content=('우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.5. 대위권 : 회사가 보험금을 지급하고 취득하는 법률상의 권리를 '
 '말합니다.<예시안내>제3자의 귀책사유로 손해가 발생한 상황에서 회사가 1,000만원의 보험금을 지급했다면, 회사는\n'
 '1,000만원에 대한 대위권만 가지며 피보험자는 제3자에 대해 1,000만원을 제외한 나머지 손해금'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000636',
              'chunk_char_len': 186,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
