from langchain_core.documents import Document

chunk = Document(
    page_content=('. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발생 신분증, 본인이 아닌 경우에는 본인의 인감증명서 또는 안전성과 신뢰성이 '
 '확보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함) 6. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 '
 '계약서(위탁관리업소 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 종류, 품종, 나이, 서비스 기간, 비용 등 포함) 7. 기타 '
 '보험수익자가 보험금의 수령에 필요하여 제출하는 서류'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 124},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000783',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
