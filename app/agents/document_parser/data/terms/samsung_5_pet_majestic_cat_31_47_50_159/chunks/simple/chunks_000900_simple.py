from langchain_core.documents import Document

chunk = Document(
    page_content=('. 12) "치아의 결손" 이란 치아의 상실 또는 발치된 경우를 말하며, 치아의 일부 손 상으로 금관치료(크라운 보철수복)를 시행한 '
 '경우에는 치아의 일부 결손을 인 정하여 1/2개 결손으로 적용한다. 13) 보철치료를 위해 발치한 정상치아, 노화로 인해 자연 발치된 '
 '치아, 보철(복합 레진, 인레이, 온레이 등)한 치아, 기존 의치(틀니, 임플란트 등)의 결손은 치아 의 상실로 인정하지 않는다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000900',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
