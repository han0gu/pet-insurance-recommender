from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 보증대<br>상 임차보증금이 3억원을 초과하는 경우는 제외한다.</p><h1 id='42' "
 "style='font-size:14px'>【소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)】</h1><br><p id='43' "
 "data-category='paragraph' style='font-size:14px'>영 제118조의4 제2항 각 호 외의 부분에서 "
 '"기획재정부령으로 정하는 것"이란 만기에</p><footer id=\'44\' style=\'font-size:14px\'>- 44 '
 "-</footer><p id='45'"),
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
