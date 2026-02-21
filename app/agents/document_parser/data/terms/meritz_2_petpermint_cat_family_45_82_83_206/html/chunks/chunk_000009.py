from langchain_core.documents import Document

chunk = Document(
    page_content=('. 진단서 상의 분류번호는 진단 당시 시행되고 있는 한국표준질병사인분류 질병코딩지침서에 따라 기재된 것을 '
 "인정합니다.</td></tr></tbody></table><br><h1 id='11' "
 "style='font-size:20px'>【한국표준질병사인분류 부호 체계】</h1><br><p id='12' "
 "data-category='paragraph' style='font-size:16px'>질병의 원인과 증상 두 가지 모두에 관한 정보를 "
 '포함<br>하는 진단을 위해 아래 두 가지 분류부호가 사용됩니<br>다'),
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
