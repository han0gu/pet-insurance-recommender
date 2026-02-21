from langchain_core.documents import Document

chunk = Document(
    page_content=('는 국외의 의료관련법에서 정한 의료기관에 입실하여 의사의 관리하에 치료에 전념하는\n'
 '것을 말합니다.제3조(보험금의 청구)# ① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.- 1. '
 '청구서(회사양식)\n'
 '- 2. 등록견의 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 미등록견의 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를 회사에 제출\n'
 '- 하시고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000668',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
