from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다.\n'
 '- 12) “치아의 결손”이란 치아의 상실 또는 발치된 경우\n'
 '- 를 말하며, 치아의 일부 손상으로 금관치료(크라운\n'
 '- 보철수복)를 시행한 경우에는 치아의 일부 결손을\n'
 '- 인정하여 1/2개 결손으로 적용한다.\n'
 '- 13) 보철치료를 위해 발치한 정상치아, 노화로 인해 자\n'
 '- 연 발치된 치아, 보철(복합레진, 인레이, 온레이\n'
 '- 등)한 치아, 기존 의치(틀니, 임플란트 등)의 결손\n'
 '- 은 치아의 상실로 인정하지 않는다.\n'
 '- 14) 상실된 치아의 크기가 크든지 또는 치간의 간격이나'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000543',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
