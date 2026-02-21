from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:16px'>지급률의 결정<br>1) 한 팔의 3대 관절 중 관절 "
 '하나에 기능장해가 생기고 다른 관절 하나에<br>기능장해가 발생한 경우 지급률은 각각 적용하여 합산한다.<br>2) 1상지(팔과 '
 "손가락)의 후유장해지급률은 원칙적으로 각각 합산하되, 지</p><br><h1 id='96' "
 "style='font-size:16px'>급률은 60% 한도로 한다.</h1><br><p id='97' "
 "data-category='list'></p><br><h1 id='98'"),
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
