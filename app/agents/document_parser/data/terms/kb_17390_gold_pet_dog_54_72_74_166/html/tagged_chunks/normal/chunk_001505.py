from langchain_core.documents import Document

chunk = Document(
    page_content=('단기 통원치료(6개월간 6회이상)</td><td>2</td></tr><tr><td>단기 통원치료(6개월간 '
 '6회미만)</td><td>0</td></tr><tr><td rowspan="3"></td><td>두 눈을 감고 일어서기 곤란하거나 두 '
 '눈을 뜨고 10m 거리를 직선으로 걷다가 쓰러지는 경우 기능</td><td>20</td></tr><tr><td>장해 두 눈을 뜨고 10m '
 '거리를 직선으로 걷다가 소견 중간에 균형을 잡으려 멈추어야 하는 경우</td><td>12</td></tr><tr><td>두 눈을 뜨고 '
 '10m 거리를 직선으로'),
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
 'indexing': {'chunk_id': 'chunk_001505',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
