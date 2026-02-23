from langchain_core.documents import Document

chunk = Document(
    page_content=('보험증권에 기재된 연간 총 보상한도액(1,000만원)을 한도\n'
 '로 합니다.# 【수의사법 제2조(정의)】이 법에서 사용하는 용어의 뜻은 다음과 같다.- 1. "수의사"란 수의업무를 담당하는 사람으로서 '
 '농림축\n'
 '- 산식품부장관의 면허를 받은 사람을 말한다.\n'
 '- 4. "동물병원"이란 동물진료업을 하는 장소로서 제17조\n'
 '- 에 따른 신고를 한 진료기관을 말한다.\n'
 '\uf000 회사가 보상하는 비용은 각 항목별 피보험자가 부담한\n'
 '치료비에서 보험증권에 기재된 자기부담금을 각각 차감한\n'
 '후, 보험증권에 기재된 보상비율(70%)을 곱한 금액을 아래'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
