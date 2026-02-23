from langchain_core.documents import Document

chunk = Document(
    page_content=('| 반려동물 | 의 보험증권에 기재된 반려동물을 말하며, 이 계약에서 가입 가능한 반려동물은 대한민국 내에서 피보험자와 거주를 함 께하고 '
 '있는 개(犬)를 말합니다. 다만 아래에 기재된 개 (犬)는 이 보험의 가입 대상이 아닙니다. 1. 보험가입 당시의 연령이 생후 60일 '
 '이하 또는 만 10세 를 초과하는 개(犬) 2. 판매점, 브리더 등이 매매(賣買)를 목적으로 사육․관 리 하는 개(犬) 3. 경찰견, '
 '구조견, 군견, 사냥개 등 특수한 목적의 개 (犬)(단, 맹도견, 청도견 등 장애인 안내견은 제외) 4'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
