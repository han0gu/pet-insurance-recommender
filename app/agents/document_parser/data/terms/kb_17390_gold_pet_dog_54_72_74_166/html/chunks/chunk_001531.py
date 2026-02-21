from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</p><br><p id='6' data-category='paragraph' "
 "style='font-size:14px'>장해의 분류 지급률<br>1) 외모에 뚜렷한 추상(추한 모습)을 남긴 때 15<br>2) 외모에 "
 "약간의 추상(추한 모습)을 남긴 때 5</p><p id='7' data-category='list' "
 "style='font-size:14px'>나.</p><br><p id='8' data-category='paragraph' "
 "style='font-size:14px'>장해판정기준</p><br><p id='9'"),
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
