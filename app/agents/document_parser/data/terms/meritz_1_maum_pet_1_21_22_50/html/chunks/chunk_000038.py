from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려동물의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할<br>수 있는 증상을 포함합니다. 다만, 보험기간 중 '
 '최초로 발견된 경우에는 해당 보험<br>기간에 한하여 보상합니다.)<br>2'),
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
