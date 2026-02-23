from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정하여 1/2개 결손으로 적용한다.\n'
 '- 13) 보철치료를 위해 발치한 정상치아, 노화로 인해 자연 발치된 치아, 보철(복합\n'
 '- 레진, 인레이, 온레이 등)한 치아, 기존 의치(틀니, 임플란트 등)의 결손은 치아\n'
 '- 의 상실로 인정하지 않는다.\n'
 '- 14) 상실된 치아의 크기가 크든지 또는 치간의 간격이나 치아 배열구조 등의 문제\n'
 '- 로 사고와 관계없이 새로운 치아가 결손된 경우에는 사고로 결손된 치아 수에\n'
 '- 따라 지급률을 결정한다.\n'
 '- 15) 어린이의 유치는 향후에 영구치로 대체되므로 후유장해의 대상이 되지 않으'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000761',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
