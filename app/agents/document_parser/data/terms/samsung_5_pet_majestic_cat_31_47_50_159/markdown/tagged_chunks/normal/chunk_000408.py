from langchain_core.documents import Document

chunk = Document(
    page_content=('확인되지 않은 사람\n'
 '∙ 병원체보유자 : 임상증상을 나타내지 않으나 감염병병원체가 확인된 사람- \n'
 '③ 향후 「감염병의 예방 및 관리에 관한 법률」 등 관계 법령에서 제외되는 감염병이 생\n'
 '기는 경우 해당 감염병은 의료법 제3조(의료기관)에 규정한 국내의 병원, 의원 또는\n'
 '국외의 의료관련법에서 정한 의료기관의 의사(치과의사는 제외합니다) 면허를 가진 자\n'
 '의 진단에 따릅니다.# 제 4조 (보험금의 청구)# ① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.1. 보험금 '
 '청구서(회사양식)\n'
 '2. 감염병 발생 신고(보고)서'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000408',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
