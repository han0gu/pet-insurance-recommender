from langchain_core.documents import Document

chunk = Document(
    page_content=('- 요구에 따르기 위하여 지출한 비용\n'
 '<용어풀이># [유익하였던 비용]물건의 개량∙이용을 위하여 지출되는 비용으로, 물건의 가치를 증가시키는 데 도움이 되는 비용을\n'
 '말합니다.# [공탁보증보험료]가압류, 가집행, 가처분 등 각종 민사사건을 신청할 때, 잘못된 신청으로 인해 발생하는 피신청인\n'
 '의 손해를 법적으로 보상해 주기 위해 법원에 납부하는 공탁금을 대신하는 보험상품을 공탁보증보'),
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
