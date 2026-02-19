from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 개별사항\n'
 '제1조 (보장의 범위)\n'
 '이 특별약관은 「상해입원수술비(당일입원제외)」 및 「상해통원수술비(외래및당일입 원)」 의 총 2개의 세부보장으로 구성되어 있습니다.\n'
 '제2조 (보험금의 지급사유)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000381',
              'chunk_char_len': 111,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
