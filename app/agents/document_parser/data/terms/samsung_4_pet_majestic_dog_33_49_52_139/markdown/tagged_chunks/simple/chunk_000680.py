from langchain_core.documents import Document

chunk = Document(
    page_content=('할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.\n'
 '제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하며, 보험금 지\n'
 '급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.# 제3조 (입원의 정의와 장소)이 특별약관에서 「입원」 이라 함은 병원 또는 '
 '의원의 의사, 치과의사 또는 한의사의 면허\n'
 '를 가진 자(이하 「의사」 라 합니다)에 의하여 질병의 치료가 필요하다고 인정된 경우로서\n'
 '자택 등에서 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의원 또'),
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
 'indexing': {'chunk_id': 'chunk_000680',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
