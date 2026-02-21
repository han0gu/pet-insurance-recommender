from langchain_core.documents import Document

chunk = Document(
    page_content=("내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 척추</p><br><p id='61' "
 "data-category='list'></p><br><p id='62' data-category='list' "
 "style='font-size:14px'>체(척추뼈 몸통)의 압박률의 합이 40% 이상일 때<br>12) ‘추간판탈출증으로 인한 심한 "
 '신경 장해’란 추간판탈출증으로 추간판을<br>2마디 이상(또는 1마디 추간판에 대해 2회 이상) 수술하고도 마미신경증<br>후군이 '
 '발생하여 하지의 현저한 마비 또는 대소변의 장해가 있는'),
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
