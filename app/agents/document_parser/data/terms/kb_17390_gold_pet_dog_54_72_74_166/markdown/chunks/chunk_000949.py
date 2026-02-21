from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- |\n'
 '| 배변· | 1) 또는 지속적인 유치도뇨관 삽입상태, 방광루, 요도루, 장루상태 2) 화장실에 가서 변기위에 앉는 일(요강을 사용하는 '
 '일 포함)과 대소변 후에 뒤처리시 다른 사람의 계속적인 도움이 필요한 상태, 또는 간헐적으로 자가 인공도뇨가 배뇨 가능한 상태(CIC), '
 '기저귀를 이용한 배뇨,배변 상태 | 15% |\n'
 '| 배변· | 3) 화장실에 가는 일, 배변, 배뇨는 독립적으로 가능하나 대소변후 뒤처리에 있어 다른 사람의 도움이 필요한 상태 | '
 '10% |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
