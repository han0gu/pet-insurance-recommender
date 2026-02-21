from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위로 인한 손해(수의사의 소견 및 처방에 의한 경우도 동일) 및 그로 인하여 가중\n'
 '- 된 손해\n'
 '13. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태# ② 회사는 아래의 의료비 및 비용 또는 손해는 '
 '보상하지 않습니다.- 1. 반려묘의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할\n'
 '- 수 있는 증상을 포함합니다. 다만 보험기간 중 최초로 발견된 경우에는 보상합니다\n'
 '- .)\n'
 '- 2. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약 · 예방 접종비용 및 정기검'),
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
