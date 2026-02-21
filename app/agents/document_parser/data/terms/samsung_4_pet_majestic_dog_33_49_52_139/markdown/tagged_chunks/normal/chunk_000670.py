from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리업소\n'
 '- 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 종류, 품종, 나이, 서비스 기간,\n'
 '- 비용 등 포함)\n'
 '- 7. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류\n'
 '② 제1항 제4호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000670',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
