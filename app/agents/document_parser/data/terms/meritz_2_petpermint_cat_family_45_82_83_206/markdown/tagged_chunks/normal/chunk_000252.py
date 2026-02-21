from langchain_core.documents import Document

chunk = Document(
    page_content=('- 용, 상담 수수료, 지도 비용 및 이와 동종의 비용\n'
 '- ⑮ 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않\n'
 '- 고 약제만 배달되는 배달료 및 이와 동종의 비용\n'
 '- ⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비\n'
 '- 등 사후에 필요한 비용\n'
 '- ⑰ 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용\n'
 '- (우송비 포함)\n'
 '112- ⑱ 과잉진료행위로 인한 비용\n'
 '- ⑲ 스케일링, 발치 등을 포함한 치아의 치과치료비용\n'
 '- (단, 치아를 제외한 구강질환 보장(구강질환의 치료\n'
 '- 목적임에도 치아에 행해지는 치료는 보장하지 않습니'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000252',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
