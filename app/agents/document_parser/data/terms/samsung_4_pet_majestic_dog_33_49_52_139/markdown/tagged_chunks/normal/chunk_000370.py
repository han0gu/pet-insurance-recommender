from langchain_core.documents import Document

chunk = Document(
    page_content=('다.- \n'
 '# 제 2조 (수술의 정의와 장소)① 이 특별약관에서「수술」이라 함은 병원 또는 의원의 의사의 면허를 가진 자(이하「의\n'
 '사」라 합니다)에 의하여「안면부 상해흉터복원」으로 치료가 필요하다고 인정된 경\n'
 '우로서 자택 등에서의 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병\n'
 '원, 의원 또는 국외의 의료관련법에서 정한 의료기관에서 의사의 관리 하에「안면부\n'
 '상해흉터복원」의 치료를 직접적인 목적으로 의료기구를 사용하여 생체(生體)에 절단'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000370',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
