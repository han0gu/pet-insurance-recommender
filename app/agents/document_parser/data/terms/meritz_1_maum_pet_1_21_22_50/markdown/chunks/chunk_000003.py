from langchain_core.documents import Document

chunk = Document(
    page_content=('- 닙니다.\n'
 '- ㉠ 판매점, 브리더 등이 매매(賣買)를 목적으로 사육ㆍ관리하는 개(犬) 또는 고양이\n'
 '- (猫)\n'
 '- ㉡ 경찰견, 구조견, 군견, 사냥개 등 특수한 목적의 개(犬)(단, 맹도견, 청도견 등\n'
 '- 장애인 안내견은 제외) 또는 특수한 목적의 고양이(猫)\n'
 '- ㉢ 투견, 경주견 등 흥행을 목적으로 사육ㆍ관리하는 개(犬) 또는 흥행을 목적으로\n'
 '- 사육ㆍ관리하는 고양이(猫)\n'
 '- ㉣ 유기동물 보호센터 등에서 사육ㆍ관리하는 개(犬) 또는 고양이(猫)'),
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
