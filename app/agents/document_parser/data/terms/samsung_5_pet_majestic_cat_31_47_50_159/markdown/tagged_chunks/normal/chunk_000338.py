from langchain_core.documents import Document

chunk = Document(
    page_content=('는 전문의를 둔 병원을 말합니다.# 제3조 (수술의 정의와 장소)① 이 특별약관에서 「수술」 이라 함은 병원 또는 의원의 의사의 면허를 '
 '가진 자(이하 「의사」 라 합니다)에 의하여 골절로 치료가 필요하다고 인정된 경우로서, 자택 등에서의\n'
 '치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의원 또는 국외의\n'
 '의료관련법에서 정한 의료기관에서 의사의 관리 하에 골절의 치료를 직접적인 목적으\n'
 '로 의료기구를 사용하여 생체(生體)에 절단(切断, 특정부위를 잘라내는 것), 절제(切除,'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000338',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
