from langchain_core.documents import Document

chunk = Document(
    page_content=('한도 내에서 아래의 권리를 가집니다. 다만, 회사가 보상한 금액이 피보험자가 입은\n'
 '손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 가\n'
 '집니다.1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권- ··· ······· 、- 89 -제3자의 '
 '귀책사유로 손해가 발생한 상황에서 회사가 1,000만원의 보험금을 지급했다면, 회사는\n'
 '1,000만원에 대한 대위권만 가지며 피보험자는 제3자에 대해 1,000만원을 제외한 나머지 손해금'),
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
