from langchain_core.documents import Document

chunk = Document(
    page_content=('그 다음달에 한하여 적용합니다.\n'
 '\uf000 회사는 이 계약의 사업방법서에서 정하는 바에 따라 운\n'
 '용자산이익률과 외부지표금리수익률을 고려하여 산출된 공\n'
 '시기준이율에 조정률을 반영하여 [보장]공시이율을 결정합\n'
 '니다.\n'
 '\uf000 [보장]공시이율의 최저보증이율은 연복리 0.3%로 합니\n'
 '다.\n'
 '\uf000 회사는 제1항부터 제3항까지의 규정에서 정한 [보장]공\n'
 '시이율을 매월 회사의 인터넷 홈페이지 등을 통해 공시합니\n'
 '다.\n'
 '\uf000 회사는 사업연도가 끝나는 날을 기준으로 1년이상 유지\n'
 '된 계약에 대하여 계약자에게 연1회이상 [보장]공시이율의\n'
 '변경내역을 통지합니다.'),
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
