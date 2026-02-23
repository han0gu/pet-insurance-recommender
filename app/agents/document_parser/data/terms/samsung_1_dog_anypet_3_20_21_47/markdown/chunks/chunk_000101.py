from langchain_core.documents import Document

chunk = Document(
    page_content=('관절탈구(고관절형성부전, 대퇴골두괴사증으로 인한 탈구 포함)를 원인으로 하여 수술을 받은 경우 수술\n'
 '당일 발생한 수술비 및 치료비를 보상하여 드립니다. 단, 보험개시일로부터 그 날을 포함하여 90일 이내\n'
 '에 발생한 손해는 보상하여 드리지 않습니다. 이 계약이 갱신계약인 경우에는 적용하지 않습니다.【수술】 동물병원의 수의사 자격을 가진 '
 "자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상해\n"
 '또는 질병 치료를 위하여 수의사법 제17조(개설)에서 규정한 국내의 동물병원에서 수의사의 관리 하에 직접적'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
