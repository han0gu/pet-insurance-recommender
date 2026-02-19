from langchain_core.documents import Document

chunk = Document(
    page_content=('. 13) 보철치료를 위해 발치한 정상치아, 노화로 인해 자 연 발치된 치아, 보철(복합레진, 인레이, 온레이 등)한 치아, 기존 '
 '의치(틀니, 임플란트 등)의 결손 은 치아의 상실로 인정하지 않는다. 14) 상실된 치아의 크기가 크든지 또는 치간의 간격이나 치아 '
 '배열구조 등의 문제로 사고와 관계없이 새로 운 치아가 결손된 경우에는 사고로 결손된 치아 수 에 따라 지급률을 결정한다. 15) 어린이의 '
 '유치는 향후에 영구치로 대체되므로 후유'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000657',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
