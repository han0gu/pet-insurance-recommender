from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 등록견의 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를 회사에 제출\n'
 '- 하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 사고증명서\n'
 '가. 이물제거 목적으로 인한 치료여부가 확인 가능한 동물병원 진료기록부\n'
 '나. 이물제거(내시경) 시행한 경우: 이물제거(내시경) 처치가 명시된 동물병원 진\n'
 '료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함), 의료비 영\n'
 '수증 및 내시경영상검사결과지 등'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000571',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
