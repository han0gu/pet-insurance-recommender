from langchain_core.documents import Document

chunk = Document(
    page_content=('해\n'
 '니다. 단, 보험기간 중에 발생한 사고로 회사가 지급하는 연간 의료비보험금의 총\n'
 '합계는 보험증권에 기재된 연간 총 보상한도액을 한도로 합니다.\n'
 '\uf000 제1항에서 정한 의료비보험금은 보험증권에 기재된 자기부담금을 차감한 후 보험\n'
 '증권에 기재된 보상비율을 곱한 금액이며 보험증권에 기재된 1일당 보상한도액을- \n'
 '# 한도로 합니다. (자기부담금은 1일당 의료비에서 차감합니다.)| 구분 | 구분 | 1일당 보상한도액 | 1일당 보상한도액 | 질 연간 '
 '총 병 보상한도액 |\n'
 '| --- | --- | --- | --- | --- |'),
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
