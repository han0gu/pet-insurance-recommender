from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비\n'
 '- 등 사후에 필요한 비용\n'
 '- ⑰ 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용\n'
 '- (우송비 포함)\n'
 '- ⑱ 과잉진료행위로 인한 비용\n'
 '\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경\n'
 '우, 제2항에 정하는 조치(마취 비용을 포함합니다.)에 대해\n'
 '서는 보험금을 지급하지 않습니다.# 제3조(수술의 정의와 장소)\uf000 이 특별약관에 있어서「수술」이라 함은 수의사가 치료\n'
 '가 필요하다고 인정한 경우로서 수의사의 관리하에 치료를'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000469',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
