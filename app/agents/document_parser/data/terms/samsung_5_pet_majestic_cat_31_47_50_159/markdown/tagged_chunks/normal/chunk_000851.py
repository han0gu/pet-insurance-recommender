from langchain_core.documents import Document

chunk = Document(
    page_content=('|  | · 빈번하고 불규칙한 배변으로 인해 2시간 이상 계속되는 업무를 수행하는 것이 어 려운 상태, 또는 배변, 배뇨는 독립적으로 '
 '가능하나 요실금, 변실금이 있는 때 (5%) |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000851',
              'chunk_char_len': 115,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
