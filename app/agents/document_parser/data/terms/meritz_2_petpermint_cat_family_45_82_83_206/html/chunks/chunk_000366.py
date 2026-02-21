from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 재가입 계약이 직전<br>계약보다 보장내용 및 범위 등이 확대된 경우 확대된 내용<br>에 대해 회사는 재가입 시점의 '
 "인수기준에 따라 승낙하거나<br>일부 보장을 제한할 수 있습니다.</p><br><p id='15' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 계약자에게 재가입주기(보장내용 "
 '변경주기)가 끝<br>나는 날 이전까지 2회 이상 재가입 요건, 보장내용 변경내<br>역, 보험료 수준, 재가입 절차 및 재가입 의사 '
 '여부를 확인<br>하는 내용 등을 서면(등기우편'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
