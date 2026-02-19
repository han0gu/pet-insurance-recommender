from langchain_core.documents import Document

chunk = Document(
    page_content=('제37조 (보험계약대출)\n'
 '① 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정한 방법에 따라 대출(이하「보 험계약대출」이라 합니다)을 받을 수 있습니다. '
 '그러나 순수보장성보험 등 보험상품의\n'
 '종류에 따라 보험계약대출이 제한될 수도 있습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 135,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
