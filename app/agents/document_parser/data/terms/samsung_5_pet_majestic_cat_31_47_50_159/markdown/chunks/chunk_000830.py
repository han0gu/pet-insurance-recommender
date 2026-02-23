from langchain_core.documents import Document

chunk = Document(
    page_content=('- 며, 노화에 의한 기능장해 또는 질병이나 외상이 없는 상태에서 예방적으로 장\n'
 '- 기를 절제, 적출한 경우는 장해로 보지 않는다.\n'
 '- 7) 상기 흉복부 및 비뇨생식기계 장해항목에 명기되지 않은 기타 장해상태에 대해\n'
 '- 서는 "<붙임> 일상생활 기본동작(ADLs) 제한 장해평가표" 에 해당하는 장해\n'
 '- 가 있을 때 ADLs 장해 지급률을 준용한다.\n'
 '# 8) 상기 장해항목에 해당되지 않는 장기간의 간병이 필요한 만성질환(만성간질환,\n'
 '만성폐쇄성폐질환 등)은 장해의 평가 대상으로 인정하지 않는다.# 13. 신경계 · 정신행동 장해-'),
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
