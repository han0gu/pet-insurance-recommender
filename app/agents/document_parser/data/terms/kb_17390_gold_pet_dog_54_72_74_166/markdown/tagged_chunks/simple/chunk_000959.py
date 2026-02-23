from langchain_core.documents import Document

chunk = Document(
    page_content=('| 기타 신체부위를 복합적으로 침범한 골절 | T02.8주) |\n'
 '| 목과 함께 머리를 침범한 탈구,염좌 및 긴장 | T03.0 |\n'
 '| 기타 신체부위를 복합적으로 침범한 탈구,염좌 및 긴장 | T03.8주) |\n'
 '| 목과 함께 머리를 침범한 으깸손상 | T04.0 |\n'
 '| 기타 신체부위를 복합적으로 침범한 으깸손상 | T04.8주) |\n'
 '| 화상 및 부식(화학약품 등에 의한 피부손상) 중, |  |\n'
 '| 머리 및 목의 화상 및 부식 (화학약품 등에 의한 피부손상) | T20 |'),
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
            'risk_domains': ['digestive', 'head', 'joint', 'skin']},
 'indexing': {'chunk_id': 'chunk_000959',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
