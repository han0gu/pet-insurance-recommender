from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 향후 「감염병의 예방 및 관리에 관한 법률」 등 관계 법령에서 제외되는 감염병이 생 기는 경우 해당 감염병은 의료법 '
 '제3조(의료기관)에 규정한 국내의 병원, 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사(치과의사는 제외합니다) 면허를 가진 자 의 '
 '진단에 따릅니다.\n'
 '제 4조 (보험금의 청구)\n'
 '① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 91},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000487',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
