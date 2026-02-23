from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 제21조(계<br>약의 무효) 제2호의 경우에는 실제 만 나이를 적용합니다.<br>\uf000 제1항의 보험나이는 계약일 현재 '
 '피보험자의 실제 만 나이를 기준으로 6개월 미만<br>의 끝수는 버리고 6개월 이상의 끝수는 1년으로 하여 계산하며, 이후 매년 계약 '
 '해<br>당일에 나이가 증가하는 것으로 합니다.<br>\uf000 청약서에 기재된 피보험자의 나이 또는 성별에 관한 사항이 신분증에 '
 '기재된 사실<br>과 다른 경우에는 신분증에 기재된 나이 또는 성별로 정정하고, 정정된 나이 또는<br>성별에 해당하는 보험금 및 '
 '보험료로 변경합니다'),
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
