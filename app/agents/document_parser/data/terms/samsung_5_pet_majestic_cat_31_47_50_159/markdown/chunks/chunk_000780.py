from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생하여 하지의 현저한 마비 또는 대소변의 장해가 있는 경우\n'
 '- 13) "추간판탈출증으로 인한 뚜렷한 신경 장해" 란 추간판탈출증으로 추간판 1마\n'
 '- 디를 수술하고도 신경생리검사에서 명확한 신경근병증의 소견이 지속되고 척\n'
 '- 추신경근의 불완전 마비가 인정되는 경우\n'
 '- 14) "추간판탈출증으로 인한 약간의 신경 장해" 란 추간판탈출증이 확인되고 신\n'
 '- 경생리검사에서 명확한 신경근병증의 소견이 지속되는 경우\n'
 '# 7. 체간골의 장해- \n'
 '# 가. 장해의 분류장 해 의 분 류 지급률(%)'),
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
