from langchain_core.documents import Document

chunk = Document(
    page_content=('르게 해석하지 않습니다.# <용어풀이># [신의성실의 원칙]계약관계의 당사자는 권리를 행사하거나 의무를 이행할 때 상대방의 정당한 이익을 '
 '배려해야 하고\n'
 '신뢰를 저버리지 않도록 행동해야 한다는 원칙을 말합니다.※ 민법 제2조(신의성실) ①권리의 행사와 의무의 이행은 신의에 좇아 성실히 '
 '하여야 한다.② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다.\n'
 '③ 회사는 보험금을 지급하지 않는 사유 등 계약자나 피보험자에게 불리하거나 부담을\n'
 '주는 내용은 확대하여 해석하지 않습니다.-'),
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
