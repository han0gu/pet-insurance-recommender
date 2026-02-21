from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'><thead><tr><td></td><td>코드 특정</td><td>질병 세부 "
 '질병명</td></tr></thead><tbody><tr><td rowspan="29"></td><td rowspan="29">805 '
 '피부질환</td><td></td></tr><tr><td>피부진균감염증(말라세치아, 사상균 '
 '등)</td></tr><tr><td>내이염</td></tr><tr><td>외이염, '
 '외이도염</td></tr><tr><td>중이염</td></tr><tr><td>개선충'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'skin']},
 'indexing': {'chunk_id': 'chunk_001817',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
