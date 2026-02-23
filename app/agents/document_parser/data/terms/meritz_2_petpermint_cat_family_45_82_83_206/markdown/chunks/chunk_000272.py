from langchain_core.documents import Document

chunk = Document(
    page_content=('감면, 사후환급금액 등을 제외한 실수납액을 말합니다)를\n'
 '이 약관에 따라 보험수익자에게 1일당 제2항에서 정한 지급\n'
 '한도 내에서 보상합니다. 다만, 연간 지급하는 총 보험금은\n'
 '보험증권에 기재된 연간 총 보상한도액(350만원)을 한도로\n'
 '합니다.# 【수의사법 제2조(정의)】이 법에서 사용하는 용어의 뜻은 다음과 같다.- 1. "수의사"란 수의업무를 담당하는 사람으로서 '
 '농림축\n'
 '- 산식품부장관의 면허를 받은 사람을 말한다.\n'
 '- 4. "동물병원"이란 동물진료업을 하는 장소로서 제17조\n'
 '- 에 따른 신고를 한 진료기관을 말한다.'),
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
