from langchain_core.documents import Document

chunk = Document(
    page_content=('| 머리 및 목의 화상 및 부식 (화학약품 등에 의한 피부손상) | T20 |\n'
 '| 눈 및 부속기에 국한된 화상 및 부식 (화학약품 등에 의한 피부손상) | T26 |\n'
 '| 동상 중, |  |\n'
 '| 머리의 표재성 동상 | T33.0 |\n'
 '| 목의 표재성 동상 | T33.1 |\n'
 '| 조직괴사를 동반한 머리의 동상 \u3000조직괴사를 동반한 목의 동상 | T34.0 T34.1 |\n'
 '|  |  |\n'
 '| 주) 1. 기타 신체부위를 복합적으로 침범한 | 사항(T00.8, T01.8, T02.8, T03.8, |'),
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
            'risk_domains': ['digestive', 'eye', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000960',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
