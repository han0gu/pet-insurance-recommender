from langchain_core.documents import Document

chunk = Document(
    page_content=('액에 대한 손해배상청구권을 가집니다.- ② 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 지키는 것\n'
 '- 에 관하여 필요한 조치를 하여야 하며, 또한 회사가 요구하는 증거 및 서류를 제출하\n'
 '- 여야 합니다. 이에 필요한 비용은 회사가 드립니다.\n'
 '- ③ 회사는 제1항 및 제2항에도 불구하고 타인을 위한 계약의 경우에는 계약자에 대한 대\n'
 '- 위권을 포기합니다.\n'
 '- ④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
