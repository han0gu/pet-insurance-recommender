from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기본공제대상자 중 장애인을 피보험자 또는 수익자로 하는 장애인전용보</p><p id='73' "
 "data-category='paragraph' style='font-size:14px'>∙</p><br><p id='74' "
 "data-category='list' style='font-size:14px'>험으로서 대통령령으로 정하는 장애인전용보장성보험료<br>2"),
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
