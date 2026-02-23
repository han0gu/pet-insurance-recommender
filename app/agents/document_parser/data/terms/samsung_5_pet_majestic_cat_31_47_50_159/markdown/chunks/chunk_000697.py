from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정질병이 악화되지 않고 유지된 경우\n'
 '⑥ 제1항의 규정에도 불구하고 다음 사항 중 어느 한 가지의 경우에 해당되는 사유로 보\n'
 '험계약에서 정한 보험금의 지급사유가 발생한 경우 회사는 보험금을 지급하여 드리며,\n'
 '보험료 납입면제사유가 발생한 경우 보험료 납입을 면제합니다.- 1. 제1항 제1호에서 지정한 특정신체부위에 발생한 질병의 합병증으로 '
 '인하여 특정신\n'
 '- 체부위 이외의 부위에 발생한 질병으로 보험계약에서 정한 보험금의 지급사유 또는\n'
 '- 보험료 납입면제사유가 발생한 경우'),
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
