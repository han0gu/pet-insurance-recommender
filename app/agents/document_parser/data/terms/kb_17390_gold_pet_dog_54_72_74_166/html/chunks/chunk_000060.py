from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 제1항에 의한 만기환급금의 지급시기가 되면 지급시기 7일 이전에 그 사유<br>통<br>와 지급할 금액을 계약자 또는 '
 '보험수익자에게 알려드리며, 만기환급금을 지급함<br>에 있어 지급일까지의 기간에 대한 이자의 계산은 "보험금을 지급할 때의 적립이율 '
 '사항<br>계산"(【별표2】참조)에 따릅니다.<br>\uf000 회사는 보험기간이 끝난 때에는 적립부분 순보험료에 대하여 보험료납입일부터 '
 '이<br>보험의 "보장성-1701 공시이율"(이하 "공시이율"이라 합니다)을 연단위 복리로 적<br>립한 금액(적립한 금액에서 '
 '중도인출액이 있었던'),
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
