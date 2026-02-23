from langchain_core.documents import Document

chunk = Document(
    page_content=('| 804 | 안과질환 | 마이보미안샘종 망막박리 |\n'
 '| 804 | 안과질환 | 망막염 / 망막변성 |\n'
 '| 804 | 안과질환 | 백내장 |\n'
 '| 804 | 안과질환 | 백내장 (저연령성) 특별 |\n'
 '| 804 | 안과질환 | 수정체 탈구 약 |\n'
 '| 804 | 안과질환 | 관 실명 |\n'
 '| 804 | 안과질환 | 안검 내 / 외번증 |\n'
 '| 804 | 안과질환 | 안검염 |\n'
 '| 804 | 안과질환 | 안방수 흐림 |\n'
 '| 804 | 안과질환 | 안와 형성 부전 별 |\n'
 '| 804 | 안과질환 | 유루증 표 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'joint']},
 'indexing': {'chunk_id': 'chunk_001039',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
