from langchain_core.documents import Document

chunk = Document(
    page_content=("id='45' data-category='list' style='font-size:16px'>⑱ 과잉진료행위로 인한 비용<br>⑲ "
 '스케일링, 발치 등을 포함한 치아의 치과치료비용<br>(단, 치아를 제외한 구강질환 보장(구강질환의 치료<br>목적임에도 치아에 행해지는 '
 '치료는 보장하지 않습니<br>다))<br>⑳ 아포퀠(Apoquel) 등의 JAK inhibitor(Janus '
 "kinase<br>inhibitor) 약물</p><br><p id='46' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000527',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
