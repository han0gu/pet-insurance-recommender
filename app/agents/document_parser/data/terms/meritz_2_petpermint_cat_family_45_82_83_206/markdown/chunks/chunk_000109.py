from langchain_core.documents import Document

chunk = Document(
    page_content=('연체된 이자(보장보험료에 대해서 평균공시이율+1%로 계산\n'
 '한 이자)를 더하여 납입하여야 합니다.# 【 부활(효력회복) 】보험료 납입을 연체하여 계약이 해지되고 계약자가 해약\n'
 '환급금을 받지 않은 경우 회사가 정하는 소정의 절차에\n'
 '따라 해지된 계약을 다시 되살리는 것을 말합니다.\uf000 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는\n'
 '제15조(계약 전 알릴 의무), 제17조(알릴 의무 위반의 효\n'
 '과), 제18조(사기에 의한 계약), 제19조(보험계약의 성립)\n'
 '및 제26조(제1회 보험료 및 회사의 보장개시)의 규정을 준'),
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
