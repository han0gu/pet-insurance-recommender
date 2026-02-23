from langchain_core.documents import Document

chunk = Document(
    page_content=('- 6. 반려견 : 보험증권에 기재된 반려견을 말하며, 이 특별약관에서 가입 가능한 반려\n'
 '- 견은 대한민국 내에서 피보험자와 거주를 함께하고 있는 개(犬)를말합니다. 다만\n'
 '- 아래에 기재된 개(犬)는 이 보험의 가입 대상이 아닙니다.\n'
 '- 가. 보험가입 당시의 연령이 생후 60일 이하 또는 만 10세를 초과하는 개(犬)\n'
 '- 나. 판매점, 브리더 등이 매매(賣買)를 목적으로 사육ㆍ관리하는 개(犬)\n'
 '- 다. 경찰견, 구조견, 군견, 사냥개 등 특수한 목적의 개(犬)(단, 맹도견, 청도견 등\n'
 '- 장애인 안내견은 제외)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
