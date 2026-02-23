from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 중도인출은 보험기간 내에 한하며, 매 보험년도마다 12회<br>에 한합니다.<br>\uf000 제1항의 중도인출의 총 누적액은 '
 '중도인출을 한번도 지급하지 않았을 경우의 기본<br>계약 해약환급금과 적립부분 해약환급금 중 적은 금액의 80%를 한도로 '
 "합니다.<br>용 어 풀 이 보험년도</p><br><table id='113' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>보험계약일로부터</td><td>다음 해의 "
 '해당일 전일까지 매1년</td><td></td><td>보험계약 단위의'),
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
