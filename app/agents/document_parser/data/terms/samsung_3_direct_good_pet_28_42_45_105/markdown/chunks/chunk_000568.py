from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑧ 피보험자에게 보험금의 지급사유 또는 보험료 납입면제사유가 발생했을 경우, 그 보험\n'
 '- 금의 지급사유 또는 보험료 납입면제사유가 특정신체부위 또는 특정질병을 직접적인\n'
 '- 원인으로 발생한 사고인가 아닌가는 의사의 진단서와 의견을 주된 판단자료로 결정합\n'
 '- 니다.\n'
 '- ⑨ 제1항의 특정신체부위와 특정질병은 4개 이내에서 선택하여 부가할 수 있습니다.\n'
 '<유의사항>회사는 제2조(특별면책조건의 내용) 제1항 각 호의 질병을 직접적인 원인으로 보험료 납입면제 사'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
