from langchain_core.documents import Document

chunk = Document(
    page_content=('. 손해의 방지 또는 경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급호송 또<br>는 그 밖의 긴급조치를 포함합니다)<br>2. '
 '제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리의 보전 또는 행사를 위<br>한 필요한 조치를 취하는 일<br>3'),
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
