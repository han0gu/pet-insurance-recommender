from langchain_core.documents import Document

chunk = Document(
    page_content=('- 여야 하며, 보험계약의 보험료를 선납하는 경우에도 또한 같습니다.\n'
 '# 제5조 (특별약관 내용의 변경)이 특별약관이 부가된 보험계약의 경우에는 보험계약 약관의 규정에도 불구하고 다음과- - 126 -\n'
 '같은 내용은 변경할 수 없습니다.1. 보험기간의 변경\n'
 '2. 감액완납보험으로의 변경- \n'
 '<용어풀이>[감액완납보험]'),
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
 'indexing': {'chunk_id': 'chunk_000683',
              'chunk_char_len': 175,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
