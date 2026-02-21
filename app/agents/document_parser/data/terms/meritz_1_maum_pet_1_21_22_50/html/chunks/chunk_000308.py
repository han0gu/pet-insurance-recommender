from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제1종 단체</h1><br><p id='52' data-category='paragraph' "
 "style='font-size:14px'>동일한 회사, 사업장, 관공서, 국영기업체, 조합 등 5인 이상의 근로자를 고용하고<br>있는 "
 '단체. 다만, 사업장, 직제, 직종 등으로 구분되어 있는 경우의 단체소속 여부는<br>관련법규 등에서 정하는 바에 '
 "따릅니다.</p><br><h1 id='53' style='font-size:14px'>2"),
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
