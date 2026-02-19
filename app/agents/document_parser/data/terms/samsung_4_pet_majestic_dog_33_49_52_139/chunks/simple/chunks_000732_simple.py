from langchain_core.documents import Document

chunk = Document(
    page_content=('. 4. 사망을 확인할 수 있는 서류(동물폐사확인서, 동물화장증명서 등) 5. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 '
 '발생 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자적 수단을 할용한 보험수익자 '
 '의사표시의 확인방법 포함)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 118},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000732',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
