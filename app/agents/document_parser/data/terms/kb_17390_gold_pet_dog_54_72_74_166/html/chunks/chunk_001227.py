from langchain_core.documents import Document

chunk = Document(
    page_content=("id='22' data-category='paragraph' style='font-size:16px'>거나 계약유지 의사를 포기하여 "
 "만기일 이전에 계약관계를 청산하는 것</p><p id='23' data-category='paragraph' "
 "style='font-size:16px'>제19조(양도)<br>보험의 목적의 양도는 회사의 서면동의 없이는 회사에 대하여 효력이 없으며, "
 '회사<br>가 서면 동의한 경우 계약으로 인하여 생긴 권리와 의무를 함께 양도한 것으로 합니<br>다'),
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
