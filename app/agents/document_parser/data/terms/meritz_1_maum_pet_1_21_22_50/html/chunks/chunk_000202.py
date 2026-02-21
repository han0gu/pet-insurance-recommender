from langchain_core.documents import Document

chunk = Document(
    page_content=('. 배상책임: 보험증권상의 보장지역 내에서 보험기간중에 발생된 보험사고로 인하여<br>타인에게 입힌 손해에 대한 법률상의 책임을 '
 '말합니다.<br>나. 장해 : 장해라 함은 신체의 상해, 질병 및 그로 인한 사망을 말합니다.<br>다. 보상한도액: 회사와 계약자간에 '
 '약정한 금액으로 피보험자가 법률상의 배상책임을<br>부담함으로써 입은 손해 중 회사가 책임지는 금액의 최대 한도를 말합니다.<br>라. '
 '자기부담금: 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부<br>담하는 일정 금액을 말합니다.<br>마'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
