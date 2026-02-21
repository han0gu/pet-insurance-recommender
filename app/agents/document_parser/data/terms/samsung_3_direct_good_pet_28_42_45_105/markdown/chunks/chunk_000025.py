from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에는 보험수익자의 청구에 따라 이미 확정된 보험금을 먼저 가지급합니다.\n'
 '- ④ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라\n'
 '- 회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.\n'
 '<용어풀이># [가지급보험금]보험금 지급이 늦어지는 경우 보험수익자 청구에 따라 확정된 보험금을 먼저 지급하는 제도\n'
 '[장해지급률]\n'
 '질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로'),
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
