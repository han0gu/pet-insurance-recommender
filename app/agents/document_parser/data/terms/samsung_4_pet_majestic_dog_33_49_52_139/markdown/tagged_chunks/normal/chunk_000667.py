from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑦ 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사는\n'
 '- 반려견 위탁비용의 전부 또는 일부를 지급하지 않습니다.\n'
 '# 제2조 (입원의 정의와 장소)이 특별약관에서 「입원」 이라 함은 병원 또는 의원의 의사, 치과의사 또는 한의사의 면허\n'
 '를 가진 자(이하 「의사」 라 합니다)에 의하여 상해의 치료가 필요하다고 인정된 경우로서\n'
 '자택 등에서 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의원 또\n'
 '는 국외의 의료관련법에서 정한 의료기관에 입실하여 의사의 관리하에 치료에 전념하는'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000667',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
