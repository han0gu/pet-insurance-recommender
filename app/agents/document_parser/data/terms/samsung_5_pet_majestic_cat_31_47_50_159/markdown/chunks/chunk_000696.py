from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 제1항 제1호 또는 제2호에서 정한 질병으로 재진단 또는 치료를 받지 않았다면 계\n'
 '- 128 -약의 청약일은 유사계약의 청약일로 봅니다.<유의사항>최초 계약 청약일부터 5년이내 재진단 또는 치료를 받고 회사에 보험금을 '
 '청구하지 않은 경우도\n'
 '재진단 또는 치료를 받은 것으로 간주합니다.# ⑤ 제4항의 재진단 또는 치료를 받지 않은 경우는 다음 각 호의 경우를 포함합니다.- 1. '
 '검진결과 추가검사 또는 치료가 필요하지 않았던 경우\n'
 '- 2. 제1항 제1호에서 정한 특정신체부위에 발생한 질병 또는 제1항 제2호에서 정한 특'),
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
