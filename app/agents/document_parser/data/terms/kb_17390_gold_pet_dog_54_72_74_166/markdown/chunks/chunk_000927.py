from langchain_core.documents import Document

chunk = Document(
    page_content=('문을 설치한 경우(치료과정에서 일시적으로 발생하는 경우는 제외)\n'
 '마) 심장기능 이상으로 인공심박동기를 영구적으로 삽입한 경우\n'
 '바) 요도괄약근 등의 기능장해로 영구적으로 인공요도괄약근을 설치한\n'
 '경우- 152 -# 5) ‘흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때’라 함은\n'
 '아래의 경우 중 하나에 해당하는 때를 말한다.\n'
 '가) 방광의 용량이 50cc 이하로 위축되었거나 요도협착, 배뇨기능 상실\n'
 '로 영구적인 간헐적 인공요도가 필요한 때\n'
 '나) 음경의 1/2 이상이 결손되었거나 질구 협착으로 성생활이 불가능한 때'),
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
