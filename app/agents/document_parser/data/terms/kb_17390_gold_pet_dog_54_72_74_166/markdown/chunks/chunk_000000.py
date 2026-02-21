from langchain_core.documents import Document

chunk = Document(
    page_content=('| 보통약관 | 보통약관 |\n'
 '| --- | --- |\n'
 '| 제1절 일반조항 | 제1절 일반조항 |\n'
 '| 제 1 관 목적 및 용어의 정의 | 제 1 관 목적 및 용어의 정의 |'),
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
