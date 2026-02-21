from langchain_core.documents import Document

chunk = Document(
    page_content=('조(해약환급금) 제4항에 따른 해약환급금을 계약자에게 지\n'
 '급합니다.\n'
 '\uf000 계약자는 제1항의 제척기간에도 불구하고 민법 등 관계\n'
 '법령에서 정하는 바에 따라 법률상의 권리를 행사 할 수 있\n'
 '습니다.# 【위법계약】금융소비자보호에 관한 법률 제47조에서 정한 적합성원\n'
 '칙, 적정성원칙, 설명의무, 불공정영업행위 금지 또는\n'
 '부당권유행위 금지를 위반한 계약을 말합니다.# 제33조(중대사유로 인한 해지)\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 '
 '날부터 1'),
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
