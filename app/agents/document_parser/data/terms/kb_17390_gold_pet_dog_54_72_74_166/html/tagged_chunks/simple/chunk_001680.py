from langchain_core.documents import Document

chunk = Document(
    page_content=('배설을 돕기 위해 설치한 의료장치나 외과적 시술물을 사용함에 있어 타인의 계속적인 도움이 필요한 또는 지속적인 유치도뇨관 삽입상태, '
 '방광루, 요도루, 장루상태</td><td></td></tr><tr><td>2) 화장실에 가서 변기위에 앉는 일(요강을 사용하는 일 '
 '포함)과 대소변 후에 뒤처리시 다른 사람의 계속적인 도움이 필요한 상태, 또는 간헐적으로 자가 인공도뇨가 배뇨 가능한 상태(CIC), '
 '기저귀를 이용한 배뇨,배변 상태</td><td>15%</td></tr><tr><td>3) 화장실에 가는 일, 배변, 배뇨는 독립적으로'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_001680',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
