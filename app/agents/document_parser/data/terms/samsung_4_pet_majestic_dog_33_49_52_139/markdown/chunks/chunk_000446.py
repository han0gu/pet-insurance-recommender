from langchain_core.documents import Document

chunk = Document(
    page_content=('- 견은 대한민국 내에서 피보험자와 거주를 함께하고 있는 개(犬)를말합니다. 다만\n'
 '- 아래에 기재된 개(犬)는 이 보험의 가입 대상이 아닙니다.\n'
 '- 가. 보험가입 당시의 연령이 생후 60일 이하 또는 만 10세를 초과하는 개(犬)\n'
 '- 나. 판매점, 브리더 등이 매매(賣買)를 목적으로 사육·관리하는 개(犬)\n'
 '- 다. 경찰견, 구조견, 군견, 사냥개 등 특수한 목적의 개(犬)(단, 맹도견, 청도견 등\n'
 '- 장애인 안내견은 제외)\n'
 '- 라. 투견, 경주견 등 흥행을 목적으로 사육·관리하는 개(犬)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
