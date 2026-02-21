from langchain_core.documents import Document

chunk = Document(
    page_content=('니다.<용어풀이>[납입기일]\n'
 '계약자가 제2회 이후의 보험료를 납입하기로 한 날을 말합니다.# 제29조 (보험료의 자동대출납입)① 계약자는 제30조(보험료의 납입이 '
 '연체되는 경우 납입최고(독촉)와 계약의 해지)에 따\n'
 '른 보험료의 납입최고(독촉)기간이 지나기 전까지 회사가 정한 방법에 따라 보험료의\n'
 '자동대출납입을 신청할 수 있으며, 이 경우 제37조(보험계약대출) 제1항에 따른 보험\n'
 '계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩니다. 다만, 계약'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
