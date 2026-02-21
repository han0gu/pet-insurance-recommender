from langchain_core.documents import Document

chunk = Document(
    page_content=("- 2. 영업일: 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일, '관공서의\n"
 "- 공휴일에 관한 규정' 에 따른 공휴일과 노동절을 제외합니다.\n"
 '# ⑤ 보험료 관련 용어1. 보험료: 손해를 보장하는데 필요한 보험료를 말합니다.# ⑥ 재가입 관련 용어- 1. 최초계약 : 최초로 '
 '체결되는 계약을 말합니다.\n'
 '- 2. 재가입계약 : 이 보험의 사업방법서에서 정한 재가입 절차에 따라 재가입된 계약을\n'
 '- 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
