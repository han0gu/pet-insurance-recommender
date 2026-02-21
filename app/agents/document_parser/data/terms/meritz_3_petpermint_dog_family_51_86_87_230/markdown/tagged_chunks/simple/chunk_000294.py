from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비\n'
 '- 등 사후에 필요한 비용\n'
 '- ⑰ 마이크로 칩 이식 비용, 각종 증빙서류의 작성비용\n'
 '- (우송비 포함)\n'
 '- ⑱ 과잉진료행위로 인한 비용\n'
 '- ⑲ 스케일링, 발치 등을 포함한 치아의 치과치료비용\n'
 '- (단, 치아를 제외한 구강질환 보장(구강질환의 치료\n'
 '- 목적임에도 치아에 행해지는 치료는 보장하지 않습니\n'
 '- 다))\n'
 '- ⑳ 아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase\n'
 '- inhibitor) 약물'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000294',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
