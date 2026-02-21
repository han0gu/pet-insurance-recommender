from langchain_core.documents import Document

chunk = Document(
    page_content=('지 사람들도 책임을 면하게 되는 것을 말합니다.| 제 3 관 계약자의 계약 전 알릴 의무 등 | 제 3 관 계약자의 계약 전 알릴 의무 '
 '등 |\n'
 '| --- | --- |'),
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
