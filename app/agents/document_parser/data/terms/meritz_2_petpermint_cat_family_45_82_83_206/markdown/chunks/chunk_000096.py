from langchain_core.documents import Document

chunk = Document(
    page_content=('기 전에 보험금 지급사유가 발생하였을 때에도 보장개시일\n'
 '부터 이 약관이 정하는 바에 따라 보장을 합니다.# 【보장개시일】회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보\n'
 '험료를 받은 날을 말하나, 회사가 승낙하기 전이라도 청약\n'
 '과 함께 제1회 보험료를 받은 경우에는 제1회 보험료를 받\n'
 '은 날을 말합니다. 또한, 보장개시일을 계약일로 봅니다.\uf000 회사는 제2항에도 불구하고 다음 중 한 가지에 해당되는\n'
 '경우에는 보장을 하지 않습니다.- ① 제15조(계약 전 알릴 의무)에 따라 계약자 또는 피보'),
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
