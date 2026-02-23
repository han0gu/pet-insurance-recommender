from langchain_core.documents import Document

chunk = Document(
    page_content=('- 되었을 때 평가하며, 객관적인 검사를 기초로 평가한다.\n'
 '- 11) 뇌·중추신경계 손상(정신·인지기능 저하, 편마비 등)으로 인한 말하는 기능의\n'
 '- 장해(실어증, 구음장애) 또는 씹어먹는 기능의 장해는 신경계·정신행동 장해\n'
 '- 평가와 비교하여 그 중 높은 지급률 하나만 인정한다.\n'
 '- 12) "치아의 결손" 이란 치아의 상실 또는 발치된 경우를 말하며, 치아의 일부 손\n'
 '- 상으로 금관치료(크라운 보철수복)를 시행한 경우에는 치아의 일부 결손을 인\n'
 '- 정하여 1/2개 결손으로 적용한다.'),
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
