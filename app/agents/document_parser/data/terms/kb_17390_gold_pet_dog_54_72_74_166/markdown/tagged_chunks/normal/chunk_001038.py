from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 804 | 안과질환 |  |\n'
 '| 804 | 안과질환 | 각막 궤양 / 미란 공 |\n'
 '| 804 | 안과질환 | 각막 위축 통 |\n'
 '| 804 | 안과질환 | 각막염 사항 |\n'
 '| 804 | 안과질환 | 건성 각결막염 |\n'
 '| 804 | 안과질환 | 결막염 |\n'
 '| 804 | 안과질환 | 녹내장 |\n'
 '| 804 | 안과질환 | 누관폐색 보 |\n'
 '| 804 | 안과질환 | 눈꼽질환 통약 |\n'
 '| 804 | 안과질환 | 마이보미안샘염 관 |\n'
 '| 804 | 안과질환 | 마이보미안샘종 망막박리 |'),
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
 'indexing': {'chunk_id': 'chunk_001038',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
