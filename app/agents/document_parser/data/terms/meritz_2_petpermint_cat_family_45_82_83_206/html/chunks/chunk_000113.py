from langchain_core.documents import Document

chunk = Document(
    page_content=('감염의 진단 확정을<br>받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성<br>립되었음을 회사가 증명하는 경우에는 계약일부터 '
 '5년 이내<br>(사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있<br>습니다.<br>\uf000 제1항에 따라 계약이 취소된 '
 "경우에는 회사는 이미 납입<br>한 보험료를 계약자에게 돌려 드립니다.</p><footer id='57' "
 "style='font-size:14px'>62</footer><h1 id='58' style='font-size:20px'>제4관 "
 '보험계약의 성립과'),
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
