from langchain_core.documents import Document

chunk = Document(
    page_content=('질병</td><td>세부 질병명</td></tr></thead><tbody><tr><td rowspan="28">804</td><td '
 'rowspan="28">안과질환</td><td></td></tr><tr><td>각막 궤양 / 미란 공</td></tr><tr><td>각막 '
 '위축 통</td></tr><tr><td>각막염 사항</td></tr><tr><td>건성 '
 '각결막염</td></tr><tr><td>결막염</td></tr><tr><td>녹내장</td></tr><tr><td>누관폐색 '
 '보</td></tr><tr><td>눈꼽질환'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001813',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
