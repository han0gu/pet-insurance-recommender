from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 있을 경우에는 그 대위권<br>\uf000 계약자 또는 피보험자는 제1항에 따라 '
 '회사가 취득한 권리를 행사하거나 지키는<br>것에 관하여 필요한 조치를 하여야 하며 또한 회사가 요구하는 증거나 서류를 제<br>출하여야 '
 '합니다.<br>\uf000 제1항 및 제2항에도 불구하고 타인을 위한 계약의 경우에는 회사는 계약자에 대<br>한 대위권을 '
 '포기합니다.<br>\uf000 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대<br>한 것인\xa0'
 '경우에는 그 권리를 취득하지 못합니다'),
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
