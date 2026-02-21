from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 적립보험료: 회사가 적립한 금액을 돌려주는데 필요한 보험료를 말합니다.\n'
 '# <용어풀이># [보험료]보험료는 계약자가 계약에 따라 회사에게 지급하여야 하는 요금을 말하며, 보험료는 「보장보험\n'
 '료」와 「적립보험료」로 구성되어 있습니다. 또한, 보험료는 보험금 지급을 위한 위험보험료, 회\n'
 '사가 적립한 금액을 돌려주기 위한 적립부분 순보험료와 부가보험료(회사 운영에 필요한 계약체결\n'
 '비용 및 계약관리비용과 보험금 지급조사를 위한 손해조사비)로 구성됩니다.- - 보험료 = 보장보험료 + 적립보험료'),
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
