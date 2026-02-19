from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 피보험자가 응급실 내원하던 도중 사망하는 경우에도 「아나필락시스」 를 직접적인 원인으로 사망한 것이 확인된 경우에는 응급실에 내원하여 '
 '진단받은 것으로 보아 제2 항을 적용합니다.\n'
 '제3조 (아나필락시스의 정의 및 진단확정)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 73},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000372',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
