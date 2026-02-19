from langchain_core.documents import Document

chunk = Document(
    page_content=('상태 (20%) · 보조기구 없이 독립적인 보행은 가능하나 보행시 파행(절뚝거림)이 있으며, 난간 을 잡지 않고는 계단을 오르고 내리기가 '
 '불가능한 상태 또는 평지에서 100m 이 상을 걷지 못하는 상태(10%)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000987',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
