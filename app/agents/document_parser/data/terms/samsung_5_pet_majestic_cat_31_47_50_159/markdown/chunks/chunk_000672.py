from langchain_core.documents import Document

chunk = Document(
    page_content=('차에 해당하지 않는 자동차- ④ 제2항 및 제3항에서 자동차관리법(하위 법령, 규칙 포함) 및 도로교통법(하위 법령, 규\n'
 '- 칙 포함) 변경시 변경된 내용을 적용합니다.\n'
 '- ⑤ 피보험자에게 보험사고가 발생했을 경우 그 사고가 이륜자동차를 운전하는 도중에 발\n'
 '- 생한 사고인가 아닌가는 관할 경찰서에서 발행한 교통사고사실 확인원 등을 주된 판\n'
 '- 단자료로 하여 결정합니다.\n'
 '<유의사항>회사는 제2조(보험금을 지급하지 않는 사유)에 해당하는 사유로 보험료 납입면제 사유가 발생한 경'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
