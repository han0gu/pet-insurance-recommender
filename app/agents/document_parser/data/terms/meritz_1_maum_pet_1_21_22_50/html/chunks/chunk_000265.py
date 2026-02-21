from langchain_core.documents import Document

chunk = Document(
    page_content=(". (이하 같습니다.)</p><p id='96' data-category='paragraph' "
 "style='font-size:14px'>제16조(알릴 의무 위반의 효과)</p><br><p id='97' "
 "data-category='paragraph' style='font-size:14px'>① 회사는 아래와 같은 사실이 있을 경우에는 "
 "손해의 발생여부에 관계없이 그 사실을 안<br>날부터 1개월 이내에 이 계약을 해지할 수 있습니다.</p><br><p id='98' "
 "data-category='list'"),
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
