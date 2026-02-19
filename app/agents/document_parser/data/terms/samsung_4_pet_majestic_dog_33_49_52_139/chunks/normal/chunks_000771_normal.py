from langchain_core.documents import Document

chunk = Document(
    page_content=('제3자의 귀책사유로 손해가 발생한 상황에서 회사가 1,000만원의 보험금을 지급했다면, 회사는 1,000만원에 대한 대위권만 가지며 '
 '피보험자는 제3자에 대해 1,000만원을 제외한 나머지 손해금 액에 대한 손해배상청구권을 가집니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 123},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000771',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
