from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경우, 제2항에 정하는 조치(마취 비용을 포함합니다.)에 대해서는 보험금을 '
 '지급하지 않습니다.\n'
 '제6조(입원의 정의와 장소)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000031',
              'chunk_char_len': 99,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
