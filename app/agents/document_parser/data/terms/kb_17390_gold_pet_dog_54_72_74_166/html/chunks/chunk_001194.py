from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자가 피해자에 대하여 부담하는 법률상의 손해배상책임액이 보험증권에</p><br><p id='213' "
 "data-category='paragraph' style='font-size:16px'>기재된 보상한도액을 명백하게 초과하는 "
 "때</p><br><p id='214' data-category='paragraph' style='font-size:16px'>2"),
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
