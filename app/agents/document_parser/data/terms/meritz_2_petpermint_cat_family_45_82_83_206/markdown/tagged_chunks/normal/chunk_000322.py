from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑮ 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않\n'
 '- 고 약제만 배달되는 배달료 및 이와 동종의 비용\n'
 '- ⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비\n'
 '- 등 사후에 필요한 비용\n'
 '- ⑰ 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용\n'
 '- (우송비 포함)\n'
 '- ⑱ 과잉진료행위로 인한 비용\n'
 '131\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경\n'
 '우, 제2항에 정하는 조치(마취 비용을 포함합니다.)에 대해'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000322',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
