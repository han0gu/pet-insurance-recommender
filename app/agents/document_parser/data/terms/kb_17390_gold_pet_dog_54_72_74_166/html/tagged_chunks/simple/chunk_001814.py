from langchain_core.documents import Document

chunk = Document(
    page_content=('보</td></tr><tr><td>눈꼽질환 통약</td></tr><tr><td>마이보미안샘염 '
 '관</td></tr><tr><td>마이보미안샘종 망막박리</td></tr><tr><td>망막염 / '
 '망막변성</td></tr><tr><td>백내장</td></tr><tr><td>백내장 (저연령성) '
 '특별</td></tr><tr><td>수정체 탈구 약</td></tr><tr><td>관 실명</td></tr><tr><td>안검 내 / '
 '외번증</td></tr><tr><td>안검염</td></tr><tr><td>안방수'),
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
 'indexing': {'chunk_id': 'chunk_001814',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
