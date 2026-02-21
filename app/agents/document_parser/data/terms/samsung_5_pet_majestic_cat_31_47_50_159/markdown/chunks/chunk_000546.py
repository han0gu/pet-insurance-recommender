from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약일 보장개시일(책임개시일)\n'
 '30일\n'
 '2022년 8월 1일 2022년 8월 31일주) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 '
 '합니\n'
 '다.<유의사항>[수술]\n'
 "동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상\n"
 '해 또는 질병 치료를 위하여 수의사법 제 17조(개설)에서 규정한 국내의 동물병원에서 수의사의관리 하에 직접적인 치료를 목적으로 기구를 '
 '사용하여 생체에 절개, 절단, 절제 등의 조작을 가하'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
