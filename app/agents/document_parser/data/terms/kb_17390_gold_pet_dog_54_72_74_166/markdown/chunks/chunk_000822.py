from langchain_core.documents import Document

chunk = Document(
    page_content=('을 무효로 하는 경우에도 제2조(특별면책(회사가 보험금을 지급하지 않는)조건의\n'
 '내용) 제1항에서 정한 특정질병에 대하여 회사가 보험금을 지급하지 않는 조건으로\n'
 '체결한 후 보장개시일 이전에 동일한 특정질병이 발생한 경우에는 계약을 무효로\n'
 '하지 않습니다.- \n'
 '제2조(특별면책(회사가 보험금을 지급하지 않는)조건의 내용)\uf000 이 특별약관에서 정한 회사가 보험금을 지급하지 않는 기간 중에 '
 '회사가지정한 질병(이하"특정질병"이라 합니다)(【별표17】(반려동물(강아지) 특정 질병 분류표))'),
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
