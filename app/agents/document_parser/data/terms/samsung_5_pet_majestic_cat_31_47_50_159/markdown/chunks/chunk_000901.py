from langchain_core.documents import Document

chunk = Document(
    page_content=('변경된 경우, 변경된 내용을 적용합니다.\n'
 '2 . 향후 「감염병의 예방 및 관리에 관한 법률」등 관계법령에서 제외되는 감염병이 생기는 경우\n'
 '해당 감염병은 신고여부와 상관없이 의사의 진단에 따릅니다.- 159 -'),
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
