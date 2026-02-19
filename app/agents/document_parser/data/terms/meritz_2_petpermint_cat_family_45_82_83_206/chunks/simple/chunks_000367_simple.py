from langchain_core.documents import Document

chunk = Document(
    page_content=('⑱ 과잉진료행위로 인한 비용 ⑲ 스케일링, 발치 등을 포함한 치아의 치과치료비용 (단, 치아를 제외한 구강질환 보장(구강질환의 치료 '
 '목적임에도 치아에 행해지는 치료는 보장하지 않습니 다)) ⑳ 아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase '
 'inhibitor) 약물\n'
 '\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경 우, 제2항에 정하는 조치(마취 비용을 포함합니다.)에 대해 서는 '
 '보험금을 지급하지 않습니다.\n'
 '제3조(수술의 정의와 장소)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 122},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000367',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
