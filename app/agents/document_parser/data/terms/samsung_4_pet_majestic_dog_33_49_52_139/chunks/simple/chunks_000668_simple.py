from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '제 5조 (보험금의 청구)\n'
 '① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '1. 보험금 청구서(회사 양식) 2. 등록견의 경우에는 동물등록증 또는 등록번호 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, '
 '측면전신사진)를 회사에 제출 하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다. 4. 사고증명서'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 112},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000668',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
