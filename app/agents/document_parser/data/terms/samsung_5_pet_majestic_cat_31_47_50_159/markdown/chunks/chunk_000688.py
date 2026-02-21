from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상 효력이 없습니다.\n'
 '- ⑥ 보험계약에서 정한 보장개시일(책임개시일) 이전에 발생한 질병에 대하여 보험계약을\n'
 '- 무효로 하는 경우에도 다음 각 호의 경우에는 보험계약을 무효로 하지 않습니다.\n'
 '- 1. 제2조(특별면책조건의 내용) 제1항 제1호의 특정신체부위에 발생한 질병에 대하여\n'
 '- 면책을 조건으로 체결한 후 보장개시일(책임개시일) 이전에 동일한 특정신체부위\n'
 '- 에 질병이 발생한 경우\n'
 '- 2. 제2조(특별면책조건의 내용) 제1항 제2호의 특정질병에 대하여 면책을 조건으로'),
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
