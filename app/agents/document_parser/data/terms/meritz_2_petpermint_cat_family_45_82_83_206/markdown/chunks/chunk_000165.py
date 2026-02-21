from langchain_core.documents import Document

chunk = Document(
    page_content=('때까지 회사는 보험금 지급지연에 따른 이자를 지급하지 않\n'
 '습니다.\uf000 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적,# 사용처 등을 명시하고 설명합니다.\uf000 보험수익자와 '
 '회사가 보험금 지급사유에 대해 합의하지\n'
 '못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제\n'
 '3자의 의견에 따를 수 있습니다. 제3자는 동물병원 소속의\n'
 '수의사 중에서 정하며, 보험금 지급사유 판정에 드는 의료\n'
 '비용은 회사가 전액 부담합니다.# 제6조(지급보험금의 계산)\uf000 동일한 반려동물과 동일한 사고에 관하여 보험금을 지급'),
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
