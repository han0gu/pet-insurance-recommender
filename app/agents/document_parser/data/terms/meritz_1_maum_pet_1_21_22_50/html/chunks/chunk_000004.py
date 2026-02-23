from langchain_core.documents import Document

chunk = Document(
    page_content=('. 진단계약: 계약을 체결하기 위하여 반려동물이 건강진단을 받아야 하는 계약을 말<br>합니다.<br>마. 피보험자: 반려동물의 소유와 '
 '관련하여 보험사고로 손해를 입은 사람(법인인 경우<br>에는 그 이사 또는 법인의 업무를 집행하는 그 밖의 기관)을 말하며, '
 '보험증권에<br>기재된 반려동물의 소유자 및 그 가족에 한합니다.<br>바. 반려동물 : 보험증권에 기재된 반려동물을 말하며, 이 '
 '계약에서 가입 가능한 반려동<br>물은 대한민국 내에서 피보험자와 거주를 함께하고 있는 개(犬) 또는 고양이(猫)를<br>말합니다'),
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
