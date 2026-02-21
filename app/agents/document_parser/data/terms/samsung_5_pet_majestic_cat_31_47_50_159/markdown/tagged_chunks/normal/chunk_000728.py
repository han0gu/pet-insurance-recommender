from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정신 또는 육체의 훼손상태 및 기능상실 상태를 말한다. 다만, 질병과 부상의 주\n'
 '- 증상과 합병증상 및 이에 대한 치료를 받는 과정에서 일시적으로 나타나는 증상\n'
 '- 은 장해에 포함되지 않는다.\n'
 '- 나. "영구적" 이라 함은 원칙적으로 치유하는 때 장래 회복할 가망이 없는 상태로서\n'
 '- 정신적 또는 육체적 훼손상태임이 의학적으로 인정되는 경우를 말한다.\n'
 '- 다. "치유된 후" 라 함은 상해 또는 질병에 대한 치료의 효과를 기대할 수 없게 되고\n'
 '- 또한 그 증상이 고정된 상태를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000728',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
