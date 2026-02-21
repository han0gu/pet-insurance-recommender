from langchain_core.documents import Document

chunk = Document(
    page_content=('복막염 또는 기타 이들과 유사한 질병 또는 상해\n'
 '5. 상병명을 알 수 없는 상해 또는 질병에 대한 치료\n'
 '6. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 정기검\n'
 '진, 예방적 검사를 위한 비용\n'
 '7. 반려동물의 임신·출산, 인공유산, 발정과 관련된 비용 및 출산 후 증상 치료비용\n'
 '8. 중성화, 불임 및 피임을 목적으로 한 처치에 따른 비용\n'
 '9. 미용으로 인한 비용\n'
 '10. 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 처치에 따른 비용'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
