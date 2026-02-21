from langchain_core.documents import Document

chunk = Document(
    page_content=('청약을 거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료\n'
 '할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '니다.3. 보상 관련 용어가. 보험가입금액: 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가\n'
 '지급할 최대 보험금을 말합니다.\n'
 '나. 자기부담금: 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부\n'
 '담하는 일정 금액을 말합니다.\n'
 '다. 보험금 분담: 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제'),
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
