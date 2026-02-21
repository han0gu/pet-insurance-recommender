from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생하여 그 치료를 직접적인 목적으로 국내에서 수의사에게 치료를 받은 때에는 피보험자가 부담\n'
 '- 한 반려동물의 치료비를 이 약관에 따라 피보험자에게 치료비보험금으로 보상하여 드립니다.\n'
 '- ② 반려동물이 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터 90일 이내의\n'
 '- 치료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 180일이내의 치료인 경우에 한합니다.'),
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
