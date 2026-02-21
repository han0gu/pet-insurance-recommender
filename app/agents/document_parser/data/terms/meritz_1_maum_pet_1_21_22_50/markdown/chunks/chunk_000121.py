from langchain_core.documents import Document

chunk = Document(
    page_content=('- 타인에게 입힌 손해에 대한 법률상의 책임을 말합니다.\n'
 '- 나. 장해 : 장해라 함은 신체의 상해, 질병 및 그로 인한 사망을 말합니다.\n'
 '- 다. 보상한도액: 회사와 계약자간에 약정한 금액으로 피보험자가 법률상의 배상책임을\n'
 '- 부담함으로써 입은 손해 중 회사가 책임지는 금액의 최대 한도를 말합니다.\n'
 '- 라. 자기부담금: 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부\n'
 '- 담하는 일정 금액을 말합니다.\n'
 '- 마. 보험금 분담: 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제'),
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
