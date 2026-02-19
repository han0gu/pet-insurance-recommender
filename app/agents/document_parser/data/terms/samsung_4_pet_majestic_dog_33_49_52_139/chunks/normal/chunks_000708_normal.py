from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험금 청구서(회사 양식) 2. 등록견의 경우에는 동물등록증 또는 등록번호 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, '
 '측면전신사진)를 회사에 제출 하시고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다. 4. 사고증명서(진료비 '
 '내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함) 및 의료비 영수증 등) 5'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 116},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000708',
              'chunk_char_len': 199,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
