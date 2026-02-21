from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출<br>납입을 신청할 경우 회사는 자동대출납입 신청내역을 서면 '
 '또는 전화(음성녹음)<br>등으로 계약자에게 알려드립니다.<br>\uf000 제1항의 규정에 의한 대출금과 보험료의 자동대출 납입일의 '
 '다음날부터 그 다음 보<br>험료의 납입최고(독촉)기간까지의 이자(보험계약대출이율 이내에서 회사가 별도로<br>정하는 이율을 적용하여 '
 '계산)를 더한 금액이 해당 보험료가 납입된 것으로 계산한<br>해약환급금과 계약자에게 지급할 기타 모든 지급금의 합계액에서 계약자의'),
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
