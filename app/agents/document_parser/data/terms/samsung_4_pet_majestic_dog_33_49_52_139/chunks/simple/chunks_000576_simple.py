from langchain_core.documents import Document

chunk = Document(
    page_content=('로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또는 보험수 익자의 책임있는 사유로 지급이 지연된 때에는 그 '
 '해당기간에 대한 이자는 더하여 지 급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000576',
              'chunk_char_len': 102,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
