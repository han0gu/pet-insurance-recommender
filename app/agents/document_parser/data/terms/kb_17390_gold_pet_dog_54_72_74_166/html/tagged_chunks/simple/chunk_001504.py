from langchain_core.documents import Document

chunk = Document(
    page_content=('내 용 검사</td><td>점수</td></tr><tr><td>양측 전정기능 소실</td><td>14</td></tr><tr><td>양측 '
 '전정기능 감소 소견</td><td>10</td></tr><tr><td rowspan="4"></td><td>일측 전정기능 소실 장기 '
 '통원치료(1년간 12회이상)</td><td>4 6</td></tr><tr><td>치료 장기 통원치료(1년간 '
 '6회이상)</td><td>4</td></tr><tr><td>병력 단기 통원치료(6개월간 '
 '6회이상)</td><td>2</td></tr><tr><td>단기'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001504',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
