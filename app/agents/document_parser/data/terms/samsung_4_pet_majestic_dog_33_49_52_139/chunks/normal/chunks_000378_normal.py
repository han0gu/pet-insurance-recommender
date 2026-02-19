from langchain_core.documents import Document

chunk = Document(
    page_content=('① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '1. 청구서(회사양식) 2. 사고증명서(응급실기록지 사본, 의사소견서 또는 진단서 등) 3. 신분증(주민등록증이나 운전면허증 등 사진이 '
 '붙은 정부기관 발행 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보 된 전자적 수단을 '
 '활용한 보험수익자 의사표시의 확인방법 포함) 4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000378',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
