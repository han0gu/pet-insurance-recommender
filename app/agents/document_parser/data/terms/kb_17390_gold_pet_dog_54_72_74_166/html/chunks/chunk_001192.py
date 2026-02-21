from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>나, 피보험자를 위하여 이러한 절차를 대행할 수 있습니다.</p><br><p id='211' "
 "data-category='list' style='font-size:14px'>\uf000 회사는 피보험자에 대하여 보상책임을 지는 한도 "
 '내에서 제1항의 절차에 협조하 질<br>거나 대행합니다'),
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
