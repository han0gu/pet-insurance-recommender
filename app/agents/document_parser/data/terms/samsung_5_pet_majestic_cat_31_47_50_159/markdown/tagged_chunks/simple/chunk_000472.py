from langchain_core.documents import Document

chunk = Document(
    page_content=('사유의 발생을 안 때에는 지체없이 그 사실을 회사에 알려야 합니다.제8조 (보험금의 청구)# ① 피보험자는 다음의 서류를 제출하고 '
 '보험금을 청구하여야 합니다.- 1. 보험금 청구서(회사 양식)\n'
 '- 2. 등록묘의 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 미등록묘의 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를 회사에 제출\n'
 '- 하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.\n'
 '- 4. 사고증명서(진료비 내역서(진료항목이 기재되어 있는 명세서, 수의사 처방전 포함)\n'
 '- 및 의료비 영수증 등)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000472',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
