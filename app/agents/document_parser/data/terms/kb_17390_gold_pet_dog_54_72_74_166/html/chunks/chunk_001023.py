from langchain_core.documents import Document

chunk = Document(
    page_content=('. 상해 또는 질병의 직접적인 치료를 목적으로 "특정약물치료Ⅱ"를 받은 경우<br>6. 상해 또는 질병의 직접적인 치료를 목적으로 '
 '"특정재활치료Ⅱ"를 받은 경우 려동<br>7'),
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
