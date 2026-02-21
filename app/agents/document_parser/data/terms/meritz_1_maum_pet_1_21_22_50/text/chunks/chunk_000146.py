from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 이에 필요한 비용은 회사가 지급합니다.\n'
 '③ 회사는 제1항 및 제2항에도 불구하고 타인을 위한 보험계약의 경우에는 계약자에 대한\n'
 '대위권을 포기합니다.\n'
 '④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한\n'
 '것인 경우에는 그 권리를 취득하지 못합니다. 다만, 손해가 그 가족의 고의로 인하여\n'
 '발생한 경우에는 그 권리를 취득합니다.제15조(계약 후 알릴 의무)① 계약을 맺은 후 보험의 목적에 아래와 같은 사실이 생긴 경우에는 '
 '계약자나 피보험자는'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
