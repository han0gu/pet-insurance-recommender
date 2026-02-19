from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다. "치유된 후" 라 함은 상해 또는 질병에 대한 치료의 효과를 기대할 수 없게 되고 또한 그 증상이 고정된 상태를 말한다. 라. '
 '다만, 영구히 고정된 증상은 아니지만 치료 종결 후 한시적으로 나타나는 장해에 대하여는 그 기간이 5년 이상인 경우 해당 장해지급률의 '
 '20%를 장해지급률로 한 다. 마'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000864',
              'chunk_char_len': 169,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
